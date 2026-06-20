#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch analysis for the craving-modulated 3-arm bandit + embedded mini-MID task
(bandit_mid_task_v12.py output).

What it does
------------
1. Reads every .csv in DATA_DIR (one file per run; reruns are separate files).
2. Splits each file into bandit trials and mini-MID ("bonus") probes by trial_type.
3. Bandit: overall and phase-wise optimal-choice accuracy across the two reversals,
   win-stay / lose-shift, reversal perseveration, choice-RT vigor, regret, exploration.
4. Mini-MID: hit rate (overall / food / neutral), target RT by cue type, premature
   and no-response rates, staircase convergence, bonus points.
5. Hybrid: post-food-probe carryover (choice RT, optimal choice, win-stay/lose-shift)
   on the bandit trial(s) following each food probe.
6. A second, group-level pass builds the across-subject incentive-vigor indices,
   including the residualized food-cued vigor measure used as the primary hybrid index.
7. A QC flagging system (qc_flags / qc_n_flags) plus a recommended-exclusion column.
8. Writes:
   - bandit_mid_subject_summary.csv   (one row per file)
   - bandit_phase_summary.csv         (subject x reversal phase, long)
   - mid_cue_summary.csv              (subject x cue type, long)
   - bandit_mid_data_dictionary.csv   (variable definitions)
   - bandit_mid_trial_cleaned.csv     (optional pooled trial-level export)

Nothing is ever deleted or overwritten in the source data. Exclusion columns are
recommendations only.

Before running
--------------
Edit only the USER SETTINGS section below.

Dependencies
------------
    python3 -m pip install pandas numpy

Run
---
    python3 analyze_bandit_mid_v1.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder containing the task CSV files (search is recursive, so the per-run
# subfolders created by the task under data/ are picked up automatically).
DATA_DIR = Path("/Users/burgerks/Desktop/bandit_mid_data")

# Where to save results. If None, saves into DATA_DIR / "analysis_output".
OUTPUT_DIR: Optional[Path] = None

# Optional: restrict to specific participant IDs. None analyzes every file.
INCLUDE_PARTICIPANTS: Optional[List[str]] = None

# Optional manual overrides if a file's participant/session metadata are wrong.
# Keys may be an exact filename or a substring of the filename.
SUBJECT_ID_MAP: Dict[str, str] = {}
SESSION_MAP: Dict[str, str] = {}

# Task design constants. These mirror the CFG block in bandit_mid_task_v12.py and
# are used to define phases and reversal windows; change them only if the task did.
N_TRIALS_EXPECTED = 200
REVERSAL_TRIALS = [69, 130]          # 1-indexed bandit trials where reversals take effect
N_BONUS_FOOD_EXPECTED = 16
N_BONUS_NEUTRAL_EXPECTED = 14
WIN_FLOOR_MS = 250                   # staircase floor (for rail detection); match the task's CFG WIN_FLOOR (250 in v13+)
WIN_CEIL_MS = 600                    # staircase ceiling (for rail detection)

# Reversal analysis windows (number of bandit trials before/after each reversal).
REVERSAL_WINDOW = 10

# Learning-curve windows: accuracy over the first/last K trials within each phase.
PHASE_EDGE_WINDOW = 15

# Choice-RT QC (seconds).
FAST_RT_THRESHOLD = 0.150
VERY_FAST_RT_THRESHOLD = 0.100

# QC / exclusion thresholds. These drive flags only; no rows are removed.
MIN_BANDIT_TRIALS_FOR_ANALYSIS = 150     # usable bandit trials required
MIN_ASYMPTOTIC_OPTIMAL = 0.45            # last-phase optimal-choice floor
MAX_LATE_CHOICE_PROP = 0.20              # share of bandit trials past the 4 s nudge
MAX_FAST_RT_PROP = 0.30                  # share of choice RTs below FAST_RT_THRESHOLD
MIN_MID_HIT_RATE = 0.40                  # staircase should hold hits near 0.667
MAX_MID_HIT_RATE = 0.90
MAX_MID_PREMATURE_PROP = 0.25            # anticipatory (too-soon) presses
MAX_MID_NO_RESPONSE_PROP = 0.20          # probes with no press at all
MIN_FOOD_PROBES_FOR_VIGOR = 8            # usable food probes for an RT-based vigor index

# Group-level (across-subject) settings.
MIN_SUBJECTS_FOR_GROUP = 8               # minimum N before z-scoring / residualizing
SAVE_TRIALWISE = True                    # write the pooled cleaned trial-level export


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


def get_manual_map_value(path: Path, mapping: Dict[str, str]) -> Optional[str]:
    """Return a manual-map value if the filename matches exactly or by substring."""
    if not mapping:
        return None
    if path.name in mapping:
        return mapping[path.name]
    for key, value in mapping.items():
        if key in path.name:
            return value
    return None


def extract_participant_id(df: pd.DataFrame, path: Path) -> str:
    """Resolve participant ID from manual map, the participant_id column, or filename."""
    manual = get_manual_map_value(path, SUBJECT_ID_MAP)
    if manual:
        return manual
    if "participant_id" in df.columns:
        val = first_nonmissing(df["participant_id"])
        if val and val.lower() != "nan":
            return val
    # The task names run folders/files sub-<pid>_ses-<ses>_<n>; pull the pid token.
    m = re.search(r"sub-([^_]+)", path.stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return path.stem


def extract_session(df: pd.DataFrame, path: Path) -> Optional[str]:
    """Resolve session label from manual map, the session column, or filename."""
    manual = get_manual_map_value(path, SESSION_MAP)
    if manual:
        return manual
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


def prop_true(x: pd.Series) -> float:
    """Proportion of True among non-missing boolean/0-1 values, NaN if none."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def prop_below(x: pd.Series, threshold: float) -> float:
    """Proportion of non-missing numeric values strictly below a threshold."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float((v < threshold).mean()) if len(v) else float("nan")


def shannon_entropy(counts: List[int]) -> float:
    """Normalized Shannon entropy (0-1) of a choice-count vector; NaN if no data."""
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return float("nan")
    p = np.array([c / total for c in counts if c > 0], dtype=float)
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(counts)))


# =============================================================================
# TRIAL EXTRACTION
# =============================================================================


def split_trials(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bandit, bonus) subframes split on trial_type, each sorted by onset.

    Bandit rows are ordered by the 1-indexed trial column; bonus rows by their
    position in the bandit stream so probe-to-trial alignment is correct.
    """
    if "trial_type" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    bandit = df[df["trial_type"] == "bandit"].copy()
    bonus = df[df["trial_type"].isin(["bonus_food", "bonus_neutral"])].copy()
    if "trial" in bandit.columns:
        bandit["trial"] = pd.to_numeric(bandit["trial"], errors="coerce")
        bandit = bandit.sort_values("trial").reset_index(drop=True)
    if "position_in_bandit_stream" in bonus.columns:
        bonus["position_in_bandit_stream"] = pd.to_numeric(
            bonus["position_in_bandit_stream"], errors="coerce")
        bonus = bonus.sort_values("position_in_bandit_stream").reset_index(drop=True)
    return bandit, bonus


def add_bandit_scoring(bandit: pd.DataFrame) -> pd.DataFrame:
    """Coerce the bandit fields used downstream and add a usable-trial flag.

    A trial is usable for behavioral scoring if it has a recorded choice and a
    valid outcome; the task waits indefinitely for a key, so true no-responses
    are rare but are still excluded here.
    """
    b = bandit.copy()
    for col in ["rt_s", "is_optimal", "points", "regret", "choice_late",
                "swap_count", "optimal_position", "choice"]:
        if col in b.columns:
            b[col] = pd.to_numeric(b[col], errors="coerce")
    b["rewarded"] = (b["outcome"].astype(str) == "reward").astype(float) if "outcome" in b.columns else np.nan
    b["usable"] = b["choice"].notna() & b["outcome"].astype(str).isin(["reward", "loss"])
    return b


def add_bonus_scoring(bonus: pd.DataFrame) -> pd.DataFrame:
    """Coerce mini-MID outcome and RT fields used downstream."""
    z = bonus.copy()
    for col in ["target_hit", "target_miss", "target_too_fast", "target_no_response",
                "target_rt_ms", "premature_rt_ms", "adaptive_window_ms",
                "food_bonus_cue", "bonus_points_earned", "position_in_bandit_stream"]:
        if col in z.columns:
            z[col] = pd.to_numeric(z[col], errors="coerce")
    return z


# =============================================================================
# BANDIT MEASURES
# =============================================================================


def phase_of_trials(bandit: pd.DataFrame) -> pd.Series:
    """Return a 1-indexed reversal phase per trial.

    Uses swap_count when present (0 before any reversal -> phase 1) because that is
    robust to config changes; otherwise falls back to the REVERSAL_TRIALS cut points.
    """
    if "swap_count" in bandit.columns and bandit["swap_count"].notna().any():
        return bandit["swap_count"].fillna(0).astype(int) + 1
    edges = [0] + REVERSAL_TRIALS + [10 ** 9]
    t = bandit["trial"]
    phase = pd.Series(1, index=bandit.index)
    for k in range(1, len(edges) - 1):
        phase = phase.mask(t >= edges[k], k + 1)
    return phase


def win_stay_lose_shift(bandit: pd.DataFrame) -> Dict[str, float]:
    """Win-stay and lose-shift over consecutive usable bandit trials.

    Identity is the chosen symbol (chosen_logo), which is fixed to a given arm for
    the whole session; stay = same symbol on the next trial. Interleaved bonus
    probes do not appear in the bandit stream, so consecutive bandit rows are adjacent.
    """
    sub = bandit[bandit["usable"]].copy().sort_values("trial").reset_index(drop=True)
    wins_repeat, wins_n, loss_switch, loss_n = 0, 0, 0, 0
    for i in range(len(sub) - 1):
        logo_now, logo_next = sub.loc[i, "chosen_logo"], sub.loc[i + 1, "chosen_logo"]
        rew = sub.loc[i, "rewarded"]
        if pd.isna(logo_now) or pd.isna(logo_next) or pd.isna(rew):
            continue
        repeated = logo_now == logo_next
        if rew == 1:
            wins_n += 1
            wins_repeat += int(repeated)
        else:
            loss_n += 1
            loss_switch += int(not repeated)
    return {
        "win_stay_prop": wins_repeat / wins_n if wins_n else float("nan"),
        "lose_shift_prop": loss_switch / loss_n if loss_n else float("nan"),
        "n_win_stay_transitions": wins_n,
        "n_lose_shift_transitions": loss_n,
    }


def reversal_metrics(bandit: pd.DataFrame) -> Dict[str, float]:
    """Pre/post accuracy and perseveration around each reversal.

    Perseveration is the share of post-reversal choices that land on the position
    that was optimal on the last pre-reversal trial; that old-best position is no
    longer optimal because each reversal rotates all three arms.
    """
    out: Dict[str, float] = {}
    w = REVERSAL_WINDOW
    persev_vals, post_acc_vals, pre_acc_vals = [], [], []
    for ri, rev in enumerate(REVERSAL_TRIALS, start=1):
        pre = bandit[(bandit["trial"] >= rev - w) & (bandit["trial"] < rev) & bandit["usable"]]
        post = bandit[(bandit["trial"] >= rev) & (bandit["trial"] < rev + w) & bandit["usable"]]
        pre_acc = prop_true(pre["is_optimal"]) if len(pre) else float("nan")
        post_acc = prop_true(post["is_optimal"]) if len(post) else float("nan")
        # Old-best position = optimal_position on the trial just before the reversal.
        prev_row = bandit[(bandit["trial"] == rev - 1)]
        old_best = float(prev_row["optimal_position"].iloc[0]) if len(prev_row) and prev_row["optimal_position"].notna().any() else float("nan")
        if not np.isnan(old_best) and len(post):
            persev = float((post["choice"] == old_best).mean())
        else:
            persev = float("nan")
        out[f"rev{ri}_pre_optimal"] = pre_acc
        out[f"rev{ri}_post_optimal"] = post_acc
        out[f"rev{ri}_perseveration"] = persev
        out[f"rev{ri}_cost"] = (pre_acc - post_acc) if pd.notna(pre_acc) and pd.notna(post_acc) else float("nan")
        for v, lst in [(persev, persev_vals), (post_acc, post_acc_vals), (pre_acc, pre_acc_vals)]:
            if pd.notna(v):
                lst.append(v)
    out["reversal_perseveration_mean"] = float(np.mean(persev_vals)) if persev_vals else float("nan")
    out["reversal_post_optimal_mean"] = float(np.mean(post_acc_vals)) if post_acc_vals else float("nan")
    out["reversal_cost_mean"] = (
        float(np.mean(pre_acc_vals) - np.mean(post_acc_vals))
        if pre_acc_vals and post_acc_vals else float("nan"))
    return out


def phase_edge_accuracy(bandit: pd.DataFrame, phase: pd.Series, edge: str, k: int) -> float:
    """Optimal-choice accuracy over the first or last k usable trials of each phase, pooled."""
    blocks = []
    for ph in sorted(phase.dropna().unique()):
        sub = bandit[(phase == ph) & bandit["usable"]].sort_values("trial")
        if not len(sub):
            continue
        blocks.append(sub.head(k) if edge == "first" else sub.tail(k))
    if not blocks:
        return float("nan")
    return prop_true(pd.concat(blocks)["is_optimal"])


def bandit_summary(bandit: pd.DataFrame) -> Dict[str, float]:
    """All within-subject bandit measures: accuracy, RT vigor, exploration, regret."""
    usable = bandit[bandit["usable"]]
    phase = phase_of_trials(bandit)
    out: Dict[str, float] = {
        "n_bandit_trials": int(len(bandit)),
        "n_bandit_usable": int(len(usable)),
        "bandit_missing_choice_prop": float((~bandit["usable"]).mean()) if len(bandit) else float("nan"),
        "max_swap_count": int(bandit["swap_count"].max()) if "swap_count" in bandit and bandit["swap_count"].notna().any() else 0,
        "overall_optimal_prop": prop_true(usable["is_optimal"]),
        "overall_reward_rate": prop_true(usable["rewarded"]),
        "total_points": float(usable["points"].sum()) if len(usable) else float("nan"),
        "mean_regret": safe_mean(usable["regret"]),
        # Choice RT is the primary implicit wanting/vigor signal within the bandit.
        "bandit_mean_choice_rt": safe_mean(usable["rt_s"]),
        "bandit_median_choice_rt": safe_median(usable["rt_s"]),
        "bandit_fast_rt_prop": prop_below(usable["rt_s"], FAST_RT_THRESHOLD),
        "bandit_very_fast_rt_prop": prop_below(usable["rt_s"], VERY_FAST_RT_THRESHOLD),
        "late_choice_prop": prop_true(bandit["choice_late"]) if "choice_late" in bandit else float("nan"),
        # First/last-phase-edge accuracy summarizes learning and post-reversal relearning.
        "phase_first_optimal_prop": phase_edge_accuracy(bandit, phase, "first", PHASE_EDGE_WINDOW),
        "phase_last_optimal_prop": phase_edge_accuracy(bandit, phase, "last", PHASE_EDGE_WINDOW),
    }
    # Asymptotic accuracy = last-phase last-edge window, used for the QC floor.
    last_phase = phase.max()
    last_sub = bandit[(phase == last_phase) & bandit["usable"]].sort_values("trial").tail(PHASE_EDGE_WINDOW)
    out["asymptotic_optimal_prop"] = prop_true(last_sub["is_optimal"]) if len(last_sub) else float("nan")
    # Exploration: switch rate and normalized entropy of the position choices.
    seq = usable.sort_values("trial")["choice"].dropna().tolist()
    out["switch_rate"] = float(np.mean([seq[i] != seq[i - 1] for i in range(1, len(seq))])) if len(seq) > 1 else float("nan")
    out["choice_entropy"] = shannon_entropy([seq.count(p) for p in (1, 2, 3)]) if seq else float("nan")
    out.update(win_stay_lose_shift(bandit))
    out.update(reversal_metrics(bandit))
    return out


# =============================================================================
# MINI-MID MEASURES
# =============================================================================


def mid_summary(bonus: pd.DataFrame) -> Dict[str, float]:
    """Mini-MID hit rate, target RT by cue type, premature/no-response, staircase state."""
    out: Dict[str, float] = {"n_bonus_trials": int(len(bonus))}
    if not len(bonus):
        return out
    food = bonus[bonus["food_bonus_cue"] == 1]
    neutral = bonus[bonus["food_bonus_cue"] == 0]
    out.update({
        "n_food_probes": int(len(food)),
        "n_neutral_probes": int(len(neutral)),
        "mid_hit_rate_overall": prop_true(bonus["target_hit"]),
        "mid_hit_rate_food": prop_true(food["target_hit"]),
        "mid_hit_rate_neutral": prop_true(neutral["target_hit"]),
        "mid_premature_prop": prop_true(bonus["target_too_fast"]),
        "mid_no_response_prop": prop_true(bonus["target_no_response"]),
        # Target RT is the raw incentive-vigor signal; food-cued RT is the key index.
        "food_mid_mean_rt_ms": safe_mean(food["target_rt_ms"]),
        "food_mid_median_rt_ms": safe_median(food["target_rt_ms"]),
        "neutral_mid_mean_rt_ms": safe_mean(neutral["target_rt_ms"]),
        "neutral_mid_median_rt_ms": safe_median(neutral["target_rt_ms"]),
        "n_food_rt_valid": int(food["target_rt_ms"].notna().sum()),
        "n_neutral_rt_valid": int(neutral["target_rt_ms"].notna().sum()),
        "bonus_points_total": float(bonus["bonus_points_earned"].sum()),
        # Final and last-five-mean windows index whether the staircase converged.
        "staircase_final_window_ms": float(bonus.sort_values("position_in_bandit_stream")["adaptive_window_ms"].dropna().iloc[-1]) if bonus["adaptive_window_ms"].notna().any() else float("nan"),
        "staircase_mean_window_ms": safe_mean(bonus["adaptive_window_ms"]),
    })
    # Version-agnostic rail detection: the window steps by a fixed amount on every
    # scored probe, so a run of consecutive identical values at the min (or max) means
    # the staircase was clamped at a bound there. The longest such run is the signature,
    # and it catches railing anywhere in the session, not just at the final probe.
    w = bonus.sort_values("position_in_bandit_stream")["adaptive_window_ms"].dropna().tolist()
    if w:
        min_w, max_w = min(w), max(w)
        def max_run(seq, val):
            best = run = 0
            for x in seq:
                run = run + 1 if x == val else 0
                best = max(best, run)
            return best
        out["staircase_min_window_ms"] = float(min_w)
        out["staircase_max_window_ms"] = float(max_w)
        out["staircase_n_at_min"] = int(sum(1 for x in w if x == min_w))
        out["staircase_n_at_max"] = int(sum(1 for x in w if x == max_w))
        out["staircase_max_run_at_min"] = max_run(w, min_w)
        out["staircase_max_run_at_max"] = max_run(w, max_w)
    # Food-minus-neutral RT contrast. Per the task design this is an individual-difference
    # readout, not a clean within-subject manipulation (there are no control trials).
    out["food_minus_neutral_rt_ms"] = (
        out["food_mid_mean_rt_ms"] - out["neutral_mid_mean_rt_ms"]
        if pd.notna(out["food_mid_mean_rt_ms"]) and pd.notna(out["neutral_mid_mean_rt_ms"]) else float("nan"))
    return out


# =============================================================================
# HYBRID (POST-PROBE CARRYOVER) MEASURES
# =============================================================================


def post_probe_metrics(bandit: pd.DataFrame, bonus: pd.DataFrame) -> Dict[str, float]:
    """Carryover onto the bandit trial(s) immediately following each probe.

    A probe with position_in_bandit_stream = p sits between bandit trial p and p+1,
    so the next bandit trial is trial == p+1. RT and optimal choice are read on that
    trial; win-stay/lose-shift compare that trial to trial p+2 conditioned on its outcome.
    """
    out: Dict[str, float] = {}
    by_trial = bandit.set_index("trial") if "trial" in bandit.columns else pd.DataFrame()

    def next_trial(p):
        return by_trial.loc[p + 1] if (p + 1) in by_trial.index else None

    for label, mask in [("food", bonus["food_bonus_cue"] == 1),
                        ("neutral", bonus["food_bonus_cue"] == 0)]:
        rts, opts = [], []
        ws_repeat = ws_n = ls_switch = ls_n = 0
        for p in bonus.loc[mask, "position_in_bandit_stream"].dropna().astype(int):
            nt = next_trial(p)
            if nt is None or not bool(nt.get("usable", False)):
                continue
            if pd.notna(nt.get("rt_s")):
                rts.append(float(nt["rt_s"]))
            if pd.notna(nt.get("is_optimal")):
                opts.append(float(nt["is_optimal"]))
            # Win-stay/lose-shift: outcome on the post-probe trial vs the trial after it.
            after = by_trial.loc[p + 2] if (p + 2) in by_trial.index else None
            if after is not None and bool(after.get("usable", False)) and pd.notna(nt.get("rewarded")):
                repeated = nt.get("chosen_logo") == after.get("chosen_logo")
                if nt["rewarded"] == 1:
                    ws_n += 1
                    ws_repeat += int(repeated)
                else:
                    ls_n += 1
                    ls_switch += int(not repeated)
        out[f"post_{label}_choice_rt"] = float(np.mean(rts)) if rts else float("nan")
        out[f"post_{label}_optimal_prop"] = float(np.mean(opts)) if opts else float("nan")
        out[f"post_{label}_n_trials"] = len(rts)
        if label == "food":
            out["post_food_win_stay"] = ws_repeat / ws_n if ws_n else float("nan")
            out["post_food_lose_shift"] = ls_switch / ls_n if ls_n else float("nan")
            out["post_food_n_win_stay"] = ws_n
            out["post_food_n_lose_shift"] = ls_n
    out["post_food_minus_neutral_choice_rt"] = (
        out.get("post_food_choice_rt", float("nan")) - out.get("post_neutral_choice_rt", float("nan"))
        if pd.notna(out.get("post_food_choice_rt", float("nan"))) and pd.notna(out.get("post_neutral_choice_rt", float("nan"))) else float("nan"))
    return out


# =============================================================================
# QC FLAGS
# =============================================================================


def build_qc_flags(s: Dict[str, float]) -> Tuple[List[str], bool, str]:
    """Return (qc_flags, recommended_exclusion, exclusion_reason) from a summary dict.

    Flags are advisory; the exclusion recommendation fires only on the conditions
    that make a participant's behavioral indices uninterpretable (too few trials,
    near-random asymptote, or a saturated/floored MID staircase).
    """
    flags: List[str] = []
    n_bandit = s.get("n_bandit_usable", 0)

    if n_bandit < MIN_BANDIT_TRIALS_FOR_ANALYSIS:
        flags.append("few_bandit_trials")
    if s.get("max_swap_count", 0) < len(REVERSAL_TRIALS):
        flags.append("reversals_not_reached")
    if pd.notna(s.get("asymptotic_optimal_prop")) and s["asymptotic_optimal_prop"] < MIN_ASYMPTOTIC_OPTIMAL:
        flags.append("low_asymptotic_accuracy")
    if pd.notna(s.get("late_choice_prop")) and s["late_choice_prop"] > MAX_LATE_CHOICE_PROP:
        flags.append("many_late_choices")
    if pd.notna(s.get("bandit_fast_rt_prop")) and s["bandit_fast_rt_prop"] > MAX_FAST_RT_PROP:
        flags.append("many_fast_choice_rts")

    hr = s.get("mid_hit_rate_overall")
    if pd.notna(hr) and hr < MIN_MID_HIT_RATE:
        flags.append("low_mid_hit_rate")
    if pd.notna(hr) and hr > MAX_MID_HIT_RATE:
        flags.append("high_mid_hit_rate")
    if pd.notna(s.get("mid_premature_prop")) and s["mid_premature_prop"] > MAX_MID_PREMATURE_PROP:
        flags.append("high_mid_premature")
    if pd.notna(s.get("mid_no_response_prop")) and s["mid_no_response_prop"] > MAX_MID_NO_RESPONSE_PROP:
        flags.append("high_mid_no_response")
    fw = s.get("staircase_final_window_ms")
    # Floored if the final window is at/below the configured floor, or if the staircase
    # was clamped at its own minimum for 3+ consecutive probes (works for any task floor).
    if pd.notna(fw) and (fw <= WIN_FLOOR_MS or s.get("staircase_max_run_at_min", 0) >= 3):
        flags.append("staircase_at_floor")
    if pd.notna(fw) and (fw >= WIN_CEIL_MS or s.get("staircase_max_run_at_max", 0) >= 3):
        flags.append("staircase_at_ceiling")
    if s.get("n_food_rt_valid", 0) < MIN_FOOD_PROBES_FOR_VIGOR:
        flags.append("few_food_probes_for_vigor")
    if s.get("n_bonus_trials", 0) < (N_BONUS_FOOD_EXPECTED + N_BONUS_NEUTRAL_EXPECTED):
        flags.append("incomplete_bonus_block")
    if s.get("duplicate_file_flag"):
        flags.append("duplicate_file")

    # Exclusion recommendation: the subset of flags that compromise interpretability.
    reasons = []
    if "few_bandit_trials" in flags:
        reasons.append("insufficient bandit trials")
    if "low_asymptotic_accuracy" in flags:
        reasons.append("near-random asymptotic choice")
    if "low_mid_hit_rate" in flags or "high_mid_hit_rate" in flags:
        reasons.append("MID staircase did not constrain hit rate")
    if "high_mid_premature" in flags:
        reasons.append("excessive anticipatory MID presses")
    recommend = len(reasons) > 0
    return flags, recommend, "; ".join(reasons)


# =============================================================================
# PER-FILE ANALYSIS
# =============================================================================


def analyze_one_file(path: Path, md5: str) -> Tuple[Dict, List[Dict]]:
    """Analyze a single run CSV; return its summary row and pooled trial rows."""
    df = pd.read_csv(path)
    pid = extract_participant_id(df, path)
    session = extract_session(df, path)
    if INCLUDE_PARTICIPANTS is not None and pid not in [str(x) for x in INCLUDE_PARTICIPANTS]:
        return {}, []

    bandit_raw, bonus_raw = split_trials(df)
    bandit = add_bandit_scoring(bandit_raw) if len(bandit_raw) else bandit_raw
    bonus = add_bonus_scoring(bonus_raw) if len(bonus_raw) else bonus_raw

    summary: Dict[str, float] = {
        "participant_id": pid,
        "session": session,
        "file_name": path.name,
        "file_path": str(path),
        "file_md5": md5,
        "task_version": first_nonmissing(df["task_version"]) if "task_version" in df.columns else None,
        "seed": first_nonmissing(df["seed"]) if "seed" in df.columns else None,
        "food_set": first_nonmissing(bonus["food_set"]) if len(bonus) and "food_set" in bonus.columns else None,
    }
    if len(bandit):
        summary.update(bandit_summary(bandit))
    if len(bonus):
        summary.update(mid_summary(bonus))
    if len(bandit) and len(bonus):
        summary.update(post_probe_metrics(bandit, bonus))

    # Pooled cleaned trial rows for the optional shareable trial-level export.
    trial_rows: List[Dict] = []
    if SAVE_TRIALWISE:
        phase = phase_of_trials(bandit) if len(bandit) else pd.Series(dtype=int)
        for i, (_, r) in enumerate(bandit.iterrows()):
            trial_rows.append({
                "participant_id": pid, "session": session, "file_name": path.name,
                "trial_type": "bandit", "trial": r.get("trial"), "phase": int(phase.iloc[i]) if len(phase) else np.nan,
                "swap_count": r.get("swap_count"), "choice": r.get("choice"), "chosen_logo": r.get("chosen_logo"),
                "rt_s": r.get("rt_s"), "outcome": r.get("outcome"), "points": r.get("points"),
                "is_optimal": r.get("is_optimal"), "regret": r.get("regret"), "usable": r.get("usable"),
            })
        for _, r in bonus.iterrows():
            trial_rows.append({
                "participant_id": pid, "session": session, "file_name": path.name,
                "trial_type": r.get("trial_type"), "position_in_bandit_stream": r.get("position_in_bandit_stream"),
                "food_bonus_cue": r.get("food_bonus_cue"), "cue_type": r.get("cue_type"),
                "target_rt_ms": r.get("target_rt_ms"), "adaptive_window_ms": r.get("adaptive_window_ms"),
                "target_hit": r.get("target_hit"), "target_too_fast": r.get("target_too_fast"),
                "target_no_response": r.get("target_no_response"),
            })
    return summary, trial_rows


def make_phase_rows(path: Path, pid: str, session: Optional[str]) -> List[Dict]:
    """Build the long per-phase accuracy/RT rows for one file (reread for clarity)."""
    df = pd.read_csv(path)
    bandit_raw, _ = split_trials(df)
    if not len(bandit_raw):
        return []
    bandit = add_bandit_scoring(bandit_raw)
    phase = phase_of_trials(bandit)
    rows = []
    for ph in sorted(phase.dropna().unique()):
        sub = bandit[(phase == ph) & bandit["usable"]]
        rows.append({
            "participant_id": pid, "session": session, "file_name": path.name,
            "phase": int(ph), "n_trials": int(len(sub)),
            "optimal_prop": prop_true(sub["is_optimal"]),
            "reward_rate": prop_true(sub["rewarded"]),
            "mean_choice_rt": safe_mean(sub["rt_s"]),
            "switch_rate": float(np.mean([sub.sort_values('trial')['choice'].tolist()[i] != sub.sort_values('trial')['choice'].tolist()[i-1] for i in range(1, len(sub))])) if len(sub) > 1 else float("nan"),
        })
    return rows


def make_cue_rows(path: Path, pid: str, session: Optional[str]) -> List[Dict]:
    """Build the long per-cue-type mini-MID rows (food vs neutral) for one file."""
    df = pd.read_csv(path)
    _, bonus_raw = split_trials(df)
    if not len(bonus_raw):
        return []
    bonus = add_bonus_scoring(bonus_raw)
    rows = []
    for label, mask in [("food", bonus["food_bonus_cue"] == 1), ("neutral", bonus["food_bonus_cue"] == 0)]:
        sub = bonus[mask]
        rows.append({
            "participant_id": pid, "session": session, "file_name": path.name,
            "cue_type": label, "n_probes": int(len(sub)),
            "hit_rate": prop_true(sub["target_hit"]),
            "mean_target_rt_ms": safe_mean(sub["target_rt_ms"]),
            "median_target_rt_ms": safe_median(sub["target_rt_ms"]),
            "premature_prop": prop_true(sub["target_too_fast"]),
            "no_response_prop": prop_true(sub["target_no_response"]),
            "n_rt_valid": int(sub["target_rt_ms"].notna().sum()),
        })
    return rows


# =============================================================================
# GROUP-LEVEL PASS (across-subject vigor indices)
# =============================================================================


def zscore(x: pd.Series) -> pd.Series:
    """Sample z-score (ddof=1) ignoring NaN; returns NaN where input is NaN."""
    v = pd.to_numeric(x, errors="coerce")
    mu, sd = v.mean(), v.std(ddof=1)
    return (v - mu) / sd if sd and not np.isnan(sd) else v * np.nan


def add_group_vigor(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add across-subject incentive-vigor indices to the subject summary.

    Builds three readouts: -z(food RT) means and medians (higher = more vigor), and
    the residualized food-cued vigor (food RT regressed on bandit median RT, MID miss
    rate, and premature rate, then sign-flipped and z-scored). The residualized index
    is the primary hybrid vigor measure. All require at least MIN_SUBJECTS_FOR_GROUP
    usable subjects; otherwise the columns are left NaN with a note printed at the end.
    """
    df = summary_df.copy()
    for c in ["food_cued_vigor_z", "food_cued_vigor_median_z", "food_cued_vigor_residualized_z"]:
        df[c] = np.nan

    eligible = df[df.get("n_food_rt_valid", 0).fillna(0) >= MIN_FOOD_PROBES_FOR_VIGOR].copy() if "n_food_rt_valid" in df else df.iloc[0:0]
    if len(eligible) < MIN_SUBJECTS_FOR_GROUP:
        df.attrs["group_note"] = (
            f"Group vigor indices not computed: only {len(eligible)} eligible subjects "
            f"(need {MIN_SUBJECTS_FOR_GROUP}).")
        return df

    # Simple sign-flipped z-scores of food-cued target RT (faster -> higher vigor).
    idx = eligible.index
    df.loc[idx, "food_cued_vigor_z"] = (-zscore(eligible["food_mid_mean_rt_ms"])).values
    df.loc[idx, "food_cued_vigor_median_z"] = (-zscore(eligible["food_mid_median_rt_ms"])).values

    # Residualize food RT on general speed and MID engagement, using complete rows only.
    pred_cols = ["bandit_median_choice_rt", "mid_no_response_prop", "mid_premature_prop"]
    have_cols = [c for c in pred_cols if c in eligible.columns]
    reg = eligible.dropna(subset=["food_mid_mean_rt_ms"] + have_cols)
    if len(reg) >= MIN_SUBJECTS_FOR_GROUP and have_cols:
        X = np.column_stack([np.ones(len(reg))] + [pd.to_numeric(reg[c], errors="coerce").values for c in have_cols])
        y = pd.to_numeric(reg["food_mid_mean_rt_ms"], errors="coerce").values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        # Sign-flip so higher = faster-than-expected (more vigor), then z-score.
        vigor = -resid
        sd = vigor.std(ddof=1)
        df.loc[reg.index, "food_cued_vigor_residualized_z"] = (vigor - vigor.mean()) / sd if sd else np.nan
    else:
        df.attrs["resid_note"] = "Residualized vigor skipped: too few complete rows."
    return df


# =============================================================================
# DATA DICTIONARY
# =============================================================================


def make_data_dictionary() -> pd.DataFrame:
    """Return the variable definitions for every output column."""
    rows = [
        ("participant_id", "Participant ID from the participant_id column, manual map, or sub-<id> in the filename."),
        ("session", "Session label from the session column, manual map, or ses-<id> in the filename."),
        ("file_name", "Source CSV filename for this run."),
        ("file_path", "Full path to the source CSV."),
        ("file_md5", "MD5 hash of the file; identical hashes flag duplicate runs."),
        ("duplicate_file_flag", "True if another analyzed file has the same MD5 hash."),
        ("duplicate_file_count", "Number of analyzed files sharing this MD5 hash."),
        ("task_version", "TASK_VERSION stamped in the data rows (provenance)."),
        ("seed", "RNG seed for the run; reproduces the bandit schedule."),
        ("food_set", "Food category shown this session: sweet, savory, or sweet+savory (between-subjects)."),
        ("n_bandit_trials", "Bandit rows present in the file."),
        ("n_bandit_usable", "Bandit trials with a recorded choice and a valid reward/loss outcome."),
        ("bandit_missing_choice_prop", "Proportion of bandit rows without a usable choice/outcome."),
        ("max_swap_count", "Highest swap_count reached; equals the number of reversals completed."),
        ("overall_optimal_prop", "Proportion of usable bandit trials choosing the current highest-probability arm."),
        ("overall_reward_rate", "Proportion of usable bandit trials that were rewarded."),
        ("total_points", "Sum of bandit points across usable trials."),
        ("mean_regret", "Mean of optimal_points minus obtained points across usable trials."),
        ("bandit_mean_choice_rt", "Mean choice reaction time (s); the primary implicit wanting/vigor signal in the bandit."),
        ("bandit_median_choice_rt", "Median choice reaction time (s); used as the general-speed covariate for residualized vigor."),
        ("bandit_fast_rt_prop", f"Proportion of choice RTs below {FAST_RT_THRESHOLD} s."),
        ("bandit_very_fast_rt_prop", f"Proportion of choice RTs below {VERY_FAST_RT_THRESHOLD} s."),
        ("late_choice_prop", "Proportion of bandit trials where the 4 s 'answer faster' nudge fired (choice_late)."),
        ("phase_first_optimal_prop", f"Optimal-choice accuracy over the first {PHASE_EDGE_WINDOW} usable trials of each phase, pooled."),
        ("phase_last_optimal_prop", f"Optimal-choice accuracy over the last {PHASE_EDGE_WINDOW} usable trials of each phase, pooled."),
        ("asymptotic_optimal_prop", f"Optimal-choice accuracy over the last {PHASE_EDGE_WINDOW} usable trials of the final phase; basis for the asymptote QC flag."),
        ("switch_rate", "Proportion of consecutive usable trials with a different position choice (exploration index)."),
        ("choice_entropy", "Normalized Shannon entropy (0-1) of position choices across usable trials."),
        ("win_stay_prop", "Proportion of rewarded usable trials whose next usable trial repeats the same symbol."),
        ("lose_shift_prop", "Proportion of non-rewarded usable trials whose next usable trial switches symbol."),
        ("n_win_stay_transitions", "Rewarded consecutive-trial transitions available for win-stay."),
        ("n_lose_shift_transitions", "Non-rewarded consecutive-trial transitions available for lose-shift."),
        ("rev1_pre_optimal", f"Optimal-choice accuracy in the {REVERSAL_WINDOW} trials before reversal 1."),
        ("rev1_post_optimal", f"Optimal-choice accuracy in the {REVERSAL_WINDOW} trials after reversal 1."),
        ("rev1_perseveration", "Proportion of post-reversal-1 choices on the position that was optimal just before reversal 1."),
        ("rev1_cost", "rev1_pre_optimal minus rev1_post_optimal (reversal accuracy drop)."),
        ("rev2_pre_optimal", f"Optimal-choice accuracy in the {REVERSAL_WINDOW} trials before reversal 2."),
        ("rev2_post_optimal", f"Optimal-choice accuracy in the {REVERSAL_WINDOW} trials after reversal 2."),
        ("rev2_perseveration", "Proportion of post-reversal-2 choices on the position that was optimal just before reversal 2."),
        ("rev2_cost", "rev2_pre_optimal minus rev2_post_optimal."),
        ("reversal_perseveration_mean", "Mean perseveration across both reversals."),
        ("reversal_post_optimal_mean", "Mean post-reversal optimal-choice accuracy across both reversals."),
        ("reversal_cost_mean", "Mean pre-reversal accuracy minus mean post-reversal accuracy across both reversals."),
        ("n_bonus_trials", "Mini-MID probe rows present (expected 30: 16 food + 14 neutral)."),
        ("n_food_probes", "Food-cue probe rows."),
        ("n_neutral_probes", "Neutral-cue probe rows."),
        ("mid_hit_rate_overall", "Proportion of probes with a target hit; the staircase targets about 0.667."),
        ("mid_hit_rate_food", "Hit rate on food-cue probes."),
        ("mid_hit_rate_neutral", "Hit rate on neutral-cue probes."),
        ("mid_premature_prop", "Proportion of probes with a too-soon (anticipatory) press."),
        ("mid_no_response_prop", "Proportion of probes with no press within the grace window."),
        ("food_mid_mean_rt_ms", "Mean target RT (ms) on food-cue probes; raw food-cued vigor signal."),
        ("food_mid_median_rt_ms", "Median target RT (ms) on food-cue probes."),
        ("neutral_mid_mean_rt_ms", "Mean target RT (ms) on neutral-cue probes."),
        ("neutral_mid_median_rt_ms", "Median target RT (ms) on neutral-cue probes."),
        ("n_food_rt_valid", "Food probes with a valid target RT (used for vigor eligibility)."),
        ("n_neutral_rt_valid", "Neutral probes with a valid target RT."),
        ("food_minus_neutral_rt_ms", "Food minus neutral mean target RT; an individual-difference contrast, NOT a within-subject manipulation (no control trials)."),
        ("bonus_points_total", "Total mini-MID bonus points earned."),
        ("staircase_final_window_ms", "Adaptive response window (ms) on the last probe; basis for floor/ceiling QC flags."),
        ("staircase_mean_window_ms", "Mean adaptive response window (ms) across probes."),
        ("staircase_min_window_ms", "Smallest adaptive window (ms) the staircase reached in this run."),
        ("staircase_max_window_ms", "Largest adaptive window (ms) the staircase reached in this run."),
        ("staircase_n_at_min", "Number of probes at the minimum window."),
        ("staircase_n_at_max", "Number of probes at the maximum window."),
        ("staircase_max_run_at_min", "Longest run of consecutive probes pinned at the minimum window; 3+ signals clamping at the floor regardless of the task's configured floor."),
        ("staircase_max_run_at_max", "Longest run of consecutive probes pinned at the maximum window; 3+ signals clamping at the ceiling."),
        ("post_food_choice_rt", "Mean bandit choice RT (s) on the usable trial immediately after each food probe."),
        ("post_neutral_choice_rt", "Mean bandit choice RT (s) on the usable trial immediately after each neutral probe."),
        ("post_food_minus_neutral_choice_rt", "post_food_choice_rt minus post_neutral_choice_rt."),
        ("post_food_optimal_prop", "Optimal-choice rate on the usable trial immediately after each food probe."),
        ("post_neutral_optimal_prop", "Optimal-choice rate on the usable trial immediately after each neutral probe."),
        ("post_food_n_trials", "Usable post-food-probe bandit trials contributing to post-food measures."),
        ("post_neutral_n_trials", "Usable post-neutral-probe bandit trials contributing to post-neutral measures."),
        ("post_food_win_stay", "Win-stay on the post-food-probe trial: repeat after a rewarded post-probe trial. Few transitions; interpret with care."),
        ("post_food_lose_shift", "Lose-shift on the post-food-probe trial: switch after a non-rewarded post-probe trial. Few transitions."),
        ("post_food_n_win_stay", "Rewarded post-food-probe transitions available for post_food_win_stay."),
        ("post_food_n_lose_shift", "Non-rewarded post-food-probe transitions available for post_food_lose_shift."),
        ("food_cued_vigor_z", "Across-subject -z of food_mid_mean_rt_ms (higher = faster food responding = more vigor)."),
        ("food_cued_vigor_median_z", "Across-subject -z of food_mid_median_rt_ms."),
        ("food_cued_vigor_residualized_z", "PRIMARY hybrid vigor index: food RT residualized on bandit median RT, MID no-response, and premature rates, sign-flipped and z-scored across subjects."),
        ("qc_flags", "Semicolon-separated QC flags from the USER SETTINGS thresholds."),
        ("qc_n_flags", "Number of QC flags for this file."),
        ("recommended_exclusion", "True if a flag compromises interpretability (few trials, near-random asymptote, broken MID staircase, or excessive anticipation)."),
        ("exclusion_reason", "Human-readable reasons behind recommended_exclusion."),
        # Long-format outputs.
        ("phase", "bandit_phase_summary / trial export: 1-indexed reversal phase (phase 1 = before any reversal)."),
        ("n_trials", "bandit_phase_summary: usable bandit trials in the phase."),
        ("optimal_prop", "bandit_phase_summary: optimal-choice accuracy within the phase."),
        ("reward_rate", "bandit_phase_summary: reward rate within the phase."),
        ("mean_choice_rt", "bandit_phase_summary: mean choice RT (s) within the phase."),
        ("cue_type", "mid_cue_summary: food or neutral."),
        ("n_probes", "mid_cue_summary: probes of this cue type."),
        ("hit_rate", "mid_cue_summary: hit rate for this cue type."),
        ("mean_target_rt_ms", "mid_cue_summary: mean target RT (ms) for this cue type."),
        ("median_target_rt_ms", "mid_cue_summary: median target RT (ms) for this cue type."),
    ]
    return pd.DataFrame(rows, columns=["variable", "definition"])


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    data_dir = DATA_DIR.expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR else data_dir / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursive search so the task's per-run subfolders under data/ are included;
    # skip any analysis outputs that may sit alongside the raw data.
    csv_files = sorted(p for p in data_dir.rglob("*.csv")
                       if not p.name.startswith(".")
                       and "analysis_output" not in p.parts
                       and not any(tag in p.name for tag in
                                   ["subject_summary", "phase_summary", "cue_summary",
                                    "data_dictionary", "trial_cleaned"]))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")

    hashes = {p: file_md5(p) for p in csv_files}

    summaries, all_trials, all_phase, all_cue = [], [], [], []
    for path in csv_files:
        try:
            summary, trial_rows = analyze_one_file(path, hashes[path])
            if not summary:
                continue
            flags_input = dict(summary)  # duplicate flag filled after counting hashes
            summaries.append((path, summary, flags_input))
            all_trials.extend(trial_rows)
            all_phase.extend(make_phase_rows(path, summary["participant_id"], summary["session"]))
            all_cue.extend(make_cue_rows(path, summary["participant_id"], summary["session"]))
        except Exception as e:
            summaries.append((path, {
                "participant_id": extract_participant_id(pd.read_csv(path, nrows=5), path),
                "file_name": path.name, "file_path": str(path), "file_md5": hashes[path],
                "qc_flags": f"ERROR: {e}", "qc_n_flags": 1, "recommended_exclusion": True,
                "exclusion_reason": f"processing error: {e}",
            }, {}))

    if not summaries:
        raise RuntimeError("No files analyzed. Check INCLUDE_PARTICIPANTS and the data folder.")

    # Duplicate-file accounting across all processed files.
    md5_counts: Dict[str, int] = {}
    for _, s, _ in summaries:
        md5_counts[s.get("file_md5")] = md5_counts.get(s.get("file_md5"), 0) + 1

    rows = []
    for path, s, _ in summaries:
        s["duplicate_file_count"] = md5_counts.get(s.get("file_md5"), 1)
        s["duplicate_file_flag"] = s["duplicate_file_count"] > 1
        if "qc_flags" not in s:  # not an error row
            flags, recommend, reason = build_qc_flags(s)
            s["qc_flags"] = ";".join(flags)
            s["qc_n_flags"] = len(flags)
            s["recommended_exclusion"] = recommend
            s["exclusion_reason"] = reason
        rows.append(s)

    summary_df = pd.DataFrame(rows)
    summary_df = add_group_vigor(summary_df)

    # Order key identifier columns first for readability.
    lead = ["participant_id", "session", "file_name", "task_version", "seed", "food_set"]
    ordered = [c for c in lead if c in summary_df.columns] + [c for c in summary_df.columns if c not in lead]
    summary_df = summary_df[ordered]

    summary_path = output_dir / "bandit_mid_subject_summary.csv"
    phase_path = output_dir / "bandit_phase_summary.csv"
    cue_path = output_dir / "mid_cue_summary.csv"
    dict_path = output_dir / "bandit_mid_data_dictionary.csv"
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(all_phase).to_csv(phase_path, index=False)
    pd.DataFrame(all_cue).to_csv(cue_path, index=False)
    make_data_dictionary().to_csv(dict_path, index=False)
    if SAVE_TRIALWISE:
        trial_path = output_dir / "bandit_mid_trial_cleaned.csv"
        pd.DataFrame(all_trials).to_csv(trial_path, index=False)

    print("Done.")
    print(f"Analyzed files: {len(summary_df)}")
    print(f"Subject summary:   {summary_path}")
    print(f"Phase summary:     {phase_path}")
    print(f"MID cue summary:   {cue_path}")
    print(f"Data dictionary:   {dict_path}")
    if SAVE_TRIALWISE:
        print(f"Trial-level export: {output_dir / 'bandit_mid_trial_cleaned.csv'}")
    if summary_df.attrs.get("group_note"):
        print("\nNote: " + summary_df.attrs["group_note"])
    if summary_df.attrs.get("resid_note"):
        print("Note: " + summary_df.attrs["resid_note"])

    flagged = summary_df[summary_df["qc_n_flags"] > 0]
    if len(flagged):
        print("\nFiles with QC flags:")
        cols = [c for c in ["participant_id", "session", "file_name", "qc_flags"] if c in flagged.columns]
        print(flagged[cols].to_string(index=False))
    excl = summary_df[summary_df.get("recommended_exclusion") == True]
    if len(excl):
        print("\nRecommended exclusions (advisory only):")
        cols = [c for c in ["participant_id", "session", "file_name", "exclusion_reason"] if c in excl.columns]
        print(excl[cols].to_string(index=False))


if __name__ == "__main__":
    main()
