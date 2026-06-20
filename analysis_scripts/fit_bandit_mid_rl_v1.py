#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-subject reinforcement-learning fits for the craving-modulated 3-arm bandit
(bandit_mid_task_v12/v13.py output). Companion to analyze_bandit_mid_v1.py.

Models
------
A three-arm Q-learning / Rescorla-Wagner agent. Each arm (screen position) carries
a value Q; only the chosen arm is updated, because participants see feedback only for
the option they picked (counterfactual outcomes are recorded but never shown). Values
carry across the two reversals with no reset, so post-reversal relearning emerges from
the update dynamics rather than being imposed.

    r_t      = +1 if rewarded, -1 if loss          (rho * r; rho fixed at 1, see note)
    PE_t     = rho * r_t - Q_chosen,t
    alpha    = alpha_pos if PE_t >= 0 else alpha_neg
    Q_chosen,t+1 = Q_chosen,t + alpha * PE_t

Choices use a 3-way softmax over arm values, with an optional stickiness/perseveration
term kappa on the previously chosen arm:

    P(choose j) proportional to exp(beta * Q_j + kappa * 1[j == prev_choice])

Three nested models are fit and compared by AIC/BIC:
    M1  single alpha, beta                      (k = 2)
    M2  alpha_pos, alpha_neg, beta              (k = 3)   <- primary
    M3  alpha_pos, alpha_neg, beta, kappa       (k = 4)

Note on rho (reward sensitivity). With a fixed reward magnitude, rho and beta are not
jointly identified from choices: only their product is. This script therefore fixes
rho = 1 by default (ESTIMATE_RHO = False) and reports beta on that scale, matching the
hierarchical fit. The between-subject vigor -> parameter links (Table 4) should be
estimated with the hierarchical partial-pooling script, not a two-stage correlation of
these point estimates, which is shrinkage-attenuated; the per-subject values here are
for QC, model comparison, starting values, and the trialwise iEEG regressors.

Outputs
-------
1. bandit_rl_subject_summary.csv     (one row per run; all three models + QC)
2. bandit_rl_trialwise_values.csv    (per-trial Q, PE, alpha_used, choice probs; M2)
3. bandit_rl_data_dictionary.csv     (variable definitions)

Nothing in the source data is modified.

Dependencies
------------
    python3 -m pip install pandas numpy scipy

Run
---
    python3 fit_bandit_mid_rl_v1.py
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder containing the task CSV files (searched recursively, like the analyzer).
DATA_DIR = Path("/Users/burgerks/Desktop/bandit_mid_data")

# Where to save results. If None, saves into DATA_DIR / "analysis_output".
OUTPUT_DIR: Optional[Path] = None

# Optional: restrict to specific participant IDs. None fits every file.
INCLUDE_PARTICIPANTS: Optional[List[str]] = None

# Reward coding for the prediction error. Symmetric gain/loss with Q0 = 0 keeps
# alpha_pos / alpha_neg interpretable as gain vs loss learning rates.
REWARD_VALUE = 1.0
LOSS_VALUE = -1.0
Q_INIT = 0.0

# Reward sensitivity rho. Fixed at 1 by default: with fixed reward magnitude rho and
# beta are confounded (only rho*beta is identified), and fixing rho=1 improves beta
# recovery. Set True only if you understand the non-identifiability; a warning is flagged.
ESTIMATE_RHO = False
RHO_FIXED = 1.0

# Optimization bounds and multistart settings.
ALPHA_BOUNDS = (0.001, 0.999)
BETA_BOUNDS = (0.01, 30.0)
KAPPA_BOUNDS = (-5.0, 5.0)
RHO_BOUNDS = (0.05, 20.0)
N_RANDOM_STARTS = 24
RANDOM_SEED = 20260620

# Minimum usable bandit trials before a fit is attempted.
MIN_TRIALS_FOR_FIT = 100

# Boundary-proximity tolerances for the "parameter railed against a bound" flag.
ALPHA_TOL = 0.005
BETA_TOL = 0.2
KAPPA_TOL = 0.05

SAVE_TRIALWISE = True                 # write per-trial Q/PE/choice-prob export (M2)


# =============================================================================
# BASIC HELPERS
# =============================================================================


def file_md5(path: Path) -> str:
    """Return the MD5 hash of a file, used to flag exact-duplicate runs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def first_nonmissing(series: Optional[pd.Series]) -> Optional[str]:
    """Return the first non-missing value of a column as a string, or None."""
    if series is None:
        return None
    s = series.dropna()
    return str(s.iloc[0]) if len(s) else None


def extract_participant_id(df: pd.DataFrame, path: Path) -> str:
    """Resolve participant ID from the participant_id column or the sub-<id> filename token."""
    if "participant_id" in df.columns:
        val = first_nonmissing(df["participant_id"])
        if val and val.lower() != "nan":
            return val
    m = re.search(r"sub-([^_]+)", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else path.stem


def extract_session(df: pd.DataFrame, path: Path) -> Optional[str]:
    """Resolve session label from the session column or the ses-<id> filename token."""
    if "session" in df.columns:
        val = first_nonmissing(df["session"])
        if val and val.lower() != "nan":
            return val
    m = re.search(r"ses-([^_]+)", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else None


def safe_mean(x: pd.Series) -> float:
    """Mean of a coerced-numeric series, NaN if empty."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def safe_median(x: pd.Series) -> float:
    """Median of a coerced-numeric series, NaN if empty."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


# =============================================================================
# TRIAL PREPARATION
# =============================================================================


def prepare_bandit_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Return usable bandit trials in presentation order, with a 0-indexed arm column.

    A trial is modelable if it has a 1..3 position choice and a reward/loss outcome.
    Bonus probes are dropped; their interleaving does not enter the value updates.
    """
    if "trial_type" not in df.columns:
        return pd.DataFrame()
    b = df[df["trial_type"] == "bandit"].copy()
    if not len(b):
        return pd.DataFrame()
    b["trial"] = pd.to_numeric(b["trial"], errors="coerce")
    b["choice"] = pd.to_numeric(b["choice"], errors="coerce")
    b = b.sort_values("trial")
    b = b[b["choice"].isin([1, 2, 3]) & b["outcome"].astype(str).isin(["reward", "loss"])].copy()
    if not len(b):
        return pd.DataFrame()
    b["arm"] = b["choice"].astype(int) - 1                       # 0,1,2 arm index
    b["reward"] = (b["outcome"].astype(str) == "reward").astype(int)
    b["rt"] = pd.to_numeric(b.get("rt_s"), errors="coerce")
    b["is_optimal"] = pd.to_numeric(b.get("is_optimal"), errors="coerce")
    b["swap_count"] = pd.to_numeric(b.get("swap_count"), errors="coerce")
    return b.reset_index(drop=True)


# =============================================================================
# MODEL CORE (shared negative log-likelihood and replay)
# =============================================================================


def _unpack(params: np.ndarray, model: str) -> Dict[str, float]:
    """Map a parameter vector to named parameters for the requested model.

    rho is appended last when ESTIMATE_RHO is True; otherwise it is held at RHO_FIXED.
    """
    p = list(params)
    out: Dict[str, float] = {}
    if model == "M1":
        out["alpha_pos"] = out["alpha_neg"] = p[0]
        out["beta"] = p[1]
        out["kappa"] = 0.0
        rest = p[2:]
    elif model == "M2":
        out["alpha_pos"], out["alpha_neg"], out["beta"] = p[0], p[1], p[2]
        out["kappa"] = 0.0
        rest = p[3:]
    else:  # M3
        out["alpha_pos"], out["alpha_neg"], out["beta"], out["kappa"] = p[0], p[1], p[2], p[3]
        rest = p[4:]
    out["rho"] = rest[0] if (ESTIMATE_RHO and rest) else RHO_FIXED
    return out


def neg_log_likelihood(params: np.ndarray, data: Tuple[np.ndarray, np.ndarray], model: str) -> float:
    """Three-arm softmax Q-learning NLL on preextracted (arms, rewards) arrays.

    Written as a scalar inner loop with an inline length-3 log-sum-exp, because the
    optimizer evaluates this thousands of times and per-trial numpy/scipy overhead dominates.
    """
    pr = _unpack(params, model)
    apos, aneg, beta, kappa, rho = pr["alpha_pos"], pr["alpha_neg"], pr["beta"], pr["kappa"], pr["rho"]
    if not (0 < apos < 1 and 0 < aneg < 1 and beta > 0 and rho > 0):
        return 1e12                                              # guard outside-bounds probes

    arms, rews = data
    q0 = q1 = q2 = Q_INIT
    prev = -1
    nll = 0.0
    for i in range(arms.shape[0]):
        c = arms[i]
        u0, u1, u2 = beta * q0, beta * q1, beta * q2
        if prev == 0:
            u0 += kappa
        elif prev == 1:
            u1 += kappa
        elif prev == 2:
            u2 += kappa
        m = u0 if (u0 >= u1 and u0 >= u2) else (u1 if u1 >= u2 else u2)
        denom = math.exp(u0 - m) + math.exp(u1 - m) + math.exp(u2 - m)
        uc = u0 if c == 0 else (u1 if c == 1 else u2)
        nll -= (uc - m - math.log(denom))                       # log P(observed choice)
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        if c == 0:
            pe = rho * r - q0
            q0 += (apos if pe >= 0 else aneg) * pe
        elif c == 1:
            pe = rho * r - q1
            q1 += (apos if pe >= 0 else aneg) * pe
        else:
            pe = rho * r - q2
            q2 += (apos if pe >= 0 else aneg) * pe
        prev = c
    return float(nll)


def replay(params: np.ndarray, trials: pd.DataFrame, model: str) -> Tuple[List[dict], np.ndarray]:
    """Replay trials at given parameters; return per-trial rows and final Q-values.

    Per-trial fields (pre-update Q, PE, alpha used, predicted choice probabilities) are
    the regressors for iEEG alignment, e.g. feedback-locked reward prediction error.
    """
    pr = _unpack(params, model)
    apos, aneg, beta, kappa, rho = pr["alpha_pos"], pr["alpha_neg"], pr["beta"], pr["kappa"], pr["rho"]
    q = np.full(3, Q_INIT, dtype=float)
    prev = -1
    rows: List[dict] = []
    eps = 1e-12
    for _, t in trials.iterrows():
        c = int(t["arm"])
        util = beta * q.copy()
        if prev >= 0:
            util[prev] += kappa
        logp = util - logsumexp(util)
        p = np.exp(logp)
        r = REWARD_VALUE if t["reward"] == 1 else LOSS_VALUE
        pe = rho * r - q[c]
        a_used = apos if pe >= 0 else aneg
        rows.append({
            "trial": int(t["trial"]) if pd.notna(t["trial"]) else np.nan,
            "swap_count": int(t["swap_count"]) if pd.notna(t["swap_count"]) else np.nan,
            "choice_arm": c + 1, "reward": int(t["reward"]), "is_optimal": t.get("is_optimal"),
            "q_arm1_pre": q[0], "q_arm2_pre": q[1], "q_arm3_pre": q[2],
            "q_chosen_pre": q[c], "p_choose_arm1": p[0], "p_choose_arm2": p[1],
            "p_choose_arm3": p[2], "p_chosen": max(min(float(p[c]), 1 - eps), eps),
            "prediction_error": pe, "alpha_used": a_used, "rt": t.get("rt"),
        })
        q[c] += a_used * pe
        prev = c
    return rows, q


# =============================================================================
# FITTING
# =============================================================================


def _start_grid(model: str, rng: np.random.Generator) -> List[np.ndarray]:
    """Build fixed plus random starting points for one model's free parameters."""
    fixed = {
        "M1": [[0.2, 3.0], [0.1, 1.0], [0.5, 5.0]],
        "M2": [[0.2, 0.2, 3.0], [0.1, 0.3, 3.0], [0.3, 0.1, 3.0], [0.05, 0.05, 1.0]],
        "M3": [[0.2, 0.2, 3.0, 0.0], [0.1, 0.3, 3.0, 0.5], [0.3, 0.1, 3.0, -0.5]],
    }[model]
    starts = [np.array(s, dtype=float) for s in fixed]
    for _ in range(N_RANDOM_STARTS):
        if model == "M1":
            s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS)]
        elif model == "M2":
            s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS)]
        else:
            s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS),
                 rng.uniform(*BETA_BOUNDS), rng.uniform(*KAPPA_BOUNDS)]
        if ESTIMATE_RHO:
            s.append(rng.uniform(*RHO_BOUNDS))
        starts.append(np.array(s, dtype=float))
    return starts


def _bounds_for(model: str) -> List[Tuple[float, float]]:
    """Optimization bounds for one model's free parameters (rho appended if estimated)."""
    base = {
        "M1": [ALPHA_BOUNDS, BETA_BOUNDS],
        "M2": [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS],
        "M3": [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS, KAPPA_BOUNDS],
    }[model]
    return base + ([RHO_BOUNDS] if ESTIMATE_RHO else [])


def fit_model(trials: pd.DataFrame, model: str) -> dict:
    """Fit one model by multistart L-BFGS-B; return parameters and fit indices."""
    n = len(trials)
    k = {"M1": 2, "M2": 3, "M3": 4}[model] + (1 if ESTIMATE_RHO else 0)
    if n < MIN_TRIALS_FOR_FIT:
        return {"model": model, "n_trials": n, "k": k, "optimizer_success": False,
                "optimizer_message": "Too few trials"}

    rng = np.random.default_rng(RANDOM_SEED + hash(model) % 1000)
    bounds = _bounds_for(model)
    data = (trials["arm"].to_numpy(), trials["reward"].to_numpy())   # preextract once for speed
    best = None
    for x0 in _start_grid(model, rng):
        res = minimize(neg_log_likelihood, x0=x0, args=(data, model),
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res

    pr = _unpack(best.x, model)
    nll = float(best.fun)
    aic = 2 * k + 2 * nll
    bic = k * math.log(n) + 2 * nll
    # McFadden pseudo-R^2 against a uniform 3-way guess; predicted-choice accuracy.
    chance = n * math.log(3.0)
    rows, _ = replay(best.x, trials, model)
    ps = np.array([r["p_chosen"] for r in rows])
    argmax_hit = np.mean([
        np.argmax([r["p_choose_arm1"], r["p_choose_arm2"], r["p_choose_arm3"]]) == (r["choice_arm"] - 1)
        for r in rows])
    return {
        "model": model, "n_trials": n, "k": k,
        "alpha_pos": pr["alpha_pos"], "alpha_neg": pr["alpha_neg"],
        "alpha_pos_minus_alpha_neg": pr["alpha_pos"] - pr["alpha_neg"],
        "beta": pr["beta"], "kappa": pr["kappa"], "rho": pr["rho"],
        "neg_log_likelihood": nll, "aic": aic, "bic": bic,
        "pseudo_r2": 1.0 - nll / chance if chance > 0 else np.nan,
        "pred_accuracy": float(argmax_hit), "mean_p_chosen": float(ps.mean()),
        "optimizer_success": bool(best.success), "optimizer_message": str(best.message),
    }


def boundary_flags(fit: dict, model: str) -> List[str]:
    """Flag free parameters sitting within tolerance of an optimization bound."""
    flags = []

    def at(name, val, lo, hi, tol):
        if val is None or pd.isna(val):
            return
        if val <= lo + tol:
            flags.append(f"{model}_{name}_at_lower_bound")
        elif val >= hi - tol:
            flags.append(f"{model}_{name}_at_upper_bound")

    if model == "M1":
        at("alpha", fit.get("alpha_pos"), *ALPHA_BOUNDS, ALPHA_TOL)
    else:
        at("alpha_pos", fit.get("alpha_pos"), *ALPHA_BOUNDS, ALPHA_TOL)
        at("alpha_neg", fit.get("alpha_neg"), *ALPHA_BOUNDS, ALPHA_TOL)
    at("beta", fit.get("beta"), *BETA_BOUNDS, BETA_TOL)
    if model == "M3":
        at("kappa", fit.get("kappa"), *KAPPA_BOUNDS, KAPPA_TOL)
    if ESTIMATE_RHO:
        at("rho", fit.get("rho"), *RHO_BOUNDS, 0.1)
    return flags


# =============================================================================
# PER-FILE ANALYSIS
# =============================================================================


def analyze_one_file(path: Path, md5: str) -> Tuple[dict, List[dict]]:
    """Fit M1/M2/M3 for one run; return the summary row and M2 trialwise rows."""
    df = pd.read_csv(path)
    pid = extract_participant_id(df, path)
    if INCLUDE_PARTICIPANTS is not None and pid not in [str(x) for x in INCLUDE_PARTICIPANTS]:
        return {}, []
    session = extract_session(df, path)
    trials = prepare_bandit_trials(df)

    summary: Dict[str, object] = {
        "participant_id": pid, "session": session, "file_name": path.name,
        "file_path": str(path), "file_md5": md5,
        "task_version": first_nonmissing(df["task_version"]) if "task_version" in df.columns else None,
        "seed": first_nonmissing(df["seed"]) if "seed" in df.columns else None,
        "n_model_trials": int(len(trials)),
        "rho_estimated": ESTIMATE_RHO,
    }
    if len(trials) < MIN_TRIALS_FOR_FIT:
        summary["optimizer_success"] = False
        summary["fit_note"] = "too few usable bandit trials"
        return summary, []

    fits = {m: fit_model(trials, m) for m in ("M1", "M2", "M3")}
    bflags = sum((boundary_flags(fits[m], m) for m in ("M1", "M2", "M3")), [])

    # Primary parameters come from M2 (dual alpha), the preregistered learning model.
    m2 = fits["M2"]
    for key in ["alpha_pos", "alpha_neg", "alpha_pos_minus_alpha_neg", "beta", "rho",
                "neg_log_likelihood", "aic", "bic", "pseudo_r2", "pred_accuracy",
                "mean_p_chosen", "optimizer_success", "optimizer_message"]:
        summary[key] = m2.get(key)
    # M3 adds stickiness; M1 is the single-alpha baseline. Keep their key fields prefixed.
    for m in ("M1", "M3"):
        for key in ["alpha_pos", "alpha_neg", "beta", "kappa", "neg_log_likelihood", "aic", "bic"]:
            summary[f"{m}_{key}"] = fits[m].get(key)

    # Model comparison: lower AIC/BIC wins. Asymmetry (alpha_pos - alpha_neg) is only
    # interpretable where M2 beats M1; stickiness only where M3 beats M2.
    summary["delta_aic_M2_minus_M1"] = m2["aic"] - fits["M1"]["aic"]
    summary["delta_bic_M2_minus_M1"] = m2["bic"] - fits["M1"]["bic"]
    summary["delta_aic_M3_minus_M2"] = fits["M3"]["aic"] - m2["aic"]
    summary["delta_bic_M3_minus_M2"] = fits["M3"]["bic"] - m2["bic"]
    aics = {m: fits[m]["aic"] for m in fits}
    bics = {m: fits[m]["bic"] for m in fits}
    summary["best_model_by_aic"] = min(aics, key=aics.get)
    summary["best_model_by_bic"] = min(bics, key=bics.get)
    summary["boundary_flags"] = ";".join(bflags)
    summary["n_boundary_flags"] = len(bflags)
    summary["rho_identifiability_note"] = (
        "rho estimated despite confounding with beta" if ESTIMATE_RHO else "rho fixed at 1")
    summary["mean_rt"] = safe_mean(trials["rt"])
    summary["median_rt"] = safe_median(trials["rt"])

    trial_rows: List[dict] = []
    if SAVE_TRIALWISE and not pd.isna(m2.get("alpha_pos", np.nan)):
        params = [m2["alpha_pos"], m2["alpha_neg"], m2["beta"]] + ([m2["rho"]] if ESTIMATE_RHO else [])
        rows, _ = replay(np.array(params), trials, "M2")
        for r in rows:
            r.update({"participant_id": pid, "session": session, "file_name": path.name})
            trial_rows.append(r)
    return summary, trial_rows


# =============================================================================
# DATA DICTIONARY
# =============================================================================


def make_data_dictionary() -> pd.DataFrame:
    """Return variable definitions for every output column."""
    rows = [
        ("participant_id", "Participant ID from the participant_id column or sub-<id> filename token."),
        ("session", "Session label from the session column or ses-<id> filename token."),
        ("file_name", "Source CSV filename for this run."),
        ("file_path", "Full path to the source CSV."),
        ("file_md5", "MD5 hash of the file (duplicate-run detection)."),
        ("task_version", "TASK_VERSION stamped in the data rows."),
        ("seed", "RNG seed for the run."),
        ("n_model_trials", "Usable bandit trials (1..3 choice and reward/loss outcome) used for fitting."),
        ("rho_estimated", "True if rho was a free parameter; False if fixed at 1 (default)."),
        ("alpha_pos", "M2 learning rate applied when the prediction error was >= 0 (gain)."),
        ("alpha_neg", "M2 learning rate applied when the prediction error was < 0 (loss)."),
        ("alpha_pos_minus_alpha_neg", "M2 asymmetry; positive means faster updating from gains. Interpret only where M2 beats M1."),
        ("beta", "M2 softmax inverse temperature; higher is more value-consistent/exploitative choice."),
        ("rho", "Reward sensitivity used in the update; fixed at 1 unless ESTIMATE_RHO is True (then weakly identified)."),
        ("neg_log_likelihood", "M2 negative log-likelihood of observed choices; lower is better."),
        ("aic", "M2 Akaike information criterion (k=3, or 4 if rho estimated)."),
        ("bic", "M2 Bayesian information criterion."),
        ("pseudo_r2", "M2 McFadden pseudo-R^2 vs a uniform 3-way guess: 1 - NLL/(n*ln3). 0 = chance, 1 = perfect."),
        ("pred_accuracy", "Proportion of trials whose observed choice is the model's argmax-probability arm (M2)."),
        ("mean_p_chosen", "Mean M2 predicted probability of the observed choice."),
        ("optimizer_success", "Whether the M2 optimizer reported convergence."),
        ("optimizer_message", "M2 optimizer message."),
        ("M1_alpha_pos", "M1 single learning rate (alpha_pos == alpha_neg)."),
        ("M1_beta", "M1 softmax inverse temperature."),
        ("M1_neg_log_likelihood", "M1 negative log-likelihood."),
        ("M1_aic", "M1 AIC (k=2)."),
        ("M1_bic", "M1 BIC (k=2)."),
        ("M3_alpha_pos", "M3 positive learning rate (model with stickiness)."),
        ("M3_alpha_neg", "M3 negative learning rate."),
        ("M3_beta", "M3 softmax inverse temperature."),
        ("M3_kappa", "M3 stickiness/perseveration on the previously chosen arm; positive = repeat regardless of value."),
        ("M3_neg_log_likelihood", "M3 negative log-likelihood."),
        ("M3_aic", "M3 AIC (k=4)."),
        ("M3_bic", "M3 BIC (k=4)."),
        ("delta_aic_M2_minus_M1", "M2 AIC minus M1 AIC. Negative => dual-alpha M2 preferred by AIC."),
        ("delta_bic_M2_minus_M1", "M2 BIC minus M1 BIC. Negative => M2 preferred by BIC."),
        ("delta_aic_M3_minus_M2", "M3 AIC minus M2 AIC. Negative => adding stickiness preferred by AIC."),
        ("delta_bic_M3_minus_M2", "M3 BIC minus M2 BIC. Negative => stickiness preferred by BIC."),
        ("best_model_by_aic", "Model with the lowest AIC among M1/M2/M3."),
        ("best_model_by_bic", "Model with the lowest BIC among M1/M2/M3."),
        ("boundary_flags", "Free parameters sitting at an optimization bound, prefixed by model; railed estimates are weakly identified."),
        ("n_boundary_flags", "Count of boundary flags across M1/M2/M3."),
        ("rho_identifiability_note", "States whether rho was fixed at 1 (recommended) or estimated despite confounding with beta."),
        ("mean_rt", "Mean choice RT (s) across modeled trials."),
        ("median_rt", "Median choice RT (s) across modeled trials."),
        # Trialwise export.
        ("trial", "Trialwise: 1-indexed bandit trial number."),
        ("swap_count", "Trialwise: reversals completed before this trial (phase = swap_count + 1)."),
        ("choice_arm", "Trialwise: chosen arm (screen position 1..3)."),
        ("reward", "Trialwise: 1 if rewarded, 0 if loss."),
        ("is_optimal", "Trialwise: 1 if the chosen arm was the current highest-probability arm."),
        ("q_arm1_pre", "Trialwise: pre-update value of arm 1 (left)."),
        ("q_arm2_pre", "Trialwise: pre-update value of arm 2 (middle)."),
        ("q_arm3_pre", "Trialwise: pre-update value of arm 3 (right)."),
        ("q_chosen_pre", "Trialwise: pre-update value of the chosen arm."),
        ("p_choose_arm1", "Trialwise: model probability of choosing arm 1."),
        ("p_choose_arm2", "Trialwise: model probability of choosing arm 2."),
        ("p_choose_arm3", "Trialwise: model probability of choosing arm 3."),
        ("p_chosen", "Trialwise: model probability of the observed choice."),
        ("prediction_error", "Trialwise: rho*r - Q_chosen; the feedback-locked RPE regressor for iEEG."),
        ("alpha_used", "Trialwise: alpha_pos or alpha_neg, by the sign of the prediction error."),
        ("rt", "Trialwise: choice reaction time (s)."),
    ]
    return pd.DataFrame(rows, columns=["variable", "definition"])


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    data_dir = DATA_DIR.expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR else data_dir / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(p for p in data_dir.rglob("*.csv")
                       if not p.name.startswith(".")
                       and "analysis_output" not in p.parts
                       and not any(tag in p.name for tag in
                                   ["subject_summary", "phase_summary", "cue_summary",
                                    "data_dictionary", "trial_cleaned", "trialwise", "rl_"]))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")

    summaries, trialwise = [], []
    for path in csv_files:
        md5 = file_md5(path)
        try:
            summary, rows = analyze_one_file(path, md5)
            if summary:
                summaries.append(summary)
                trialwise.extend(rows)
        except Exception as e:
            summaries.append({"participant_id": extract_participant_id(pd.read_csv(path, nrows=5), path),
                              "file_name": path.name, "file_path": str(path), "file_md5": md5,
                              "optimizer_success": False, "fit_note": f"ERROR: {e}"})

    if not summaries:
        raise RuntimeError("No files fitted. Check INCLUDE_PARTICIPANTS and the data folder.")

    summary_df = pd.DataFrame(summaries)
    dup = summary_df["file_md5"].value_counts().to_dict()
    summary_df["duplicate_file_flag"] = summary_df["file_md5"].map(dup) > 1
    lead = ["participant_id", "session", "file_name", "task_version", "seed", "n_model_trials"]
    summary_df = summary_df[[c for c in lead if c in summary_df.columns] +
                            [c for c in summary_df.columns if c not in lead]]

    summary_path = output_dir / "bandit_rl_subject_summary.csv"
    dict_path = output_dir / "bandit_rl_data_dictionary.csv"
    summary_df.to_csv(summary_path, index=False)
    make_data_dictionary().to_csv(dict_path, index=False)
    if SAVE_TRIALWISE:
        trial_path = output_dir / "bandit_rl_trialwise_values.csv"
        pd.DataFrame(trialwise).to_csv(trial_path, index=False)

    print("Done.")
    print(f"Fitted files: {len(summary_df)}")
    print(f"Subject summary: {summary_path}")
    print(f"Data dictionary: {dict_path}")
    if SAVE_TRIALWISE:
        print(f"Trialwise values: {output_dir / 'bandit_rl_trialwise_values.csv'}")

    bad = summary_df[summary_df.get("optimizer_success") != True]
    if len(bad):
        print("\nFiles with fit issues:")
        cols = [c for c in ["participant_id", "session", "file_name", "fit_note", "optimizer_message"] if c in bad.columns]
        print(bad[cols].to_string(index=False))
    if "n_boundary_flags" in summary_df.columns:
        railed = summary_df[summary_df["n_boundary_flags"].fillna(0) > 0]
        if len(railed):
            print("\nFiles with boundary parameters (interpret those parameters with caution):")
            cols = [c for c in ["participant_id", "session", "file_name", "boundary_flags"] if c in railed.columns]
            print(railed[cols].to_string(index=False))


if __name__ == "__main__":
    main()
