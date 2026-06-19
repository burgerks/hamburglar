"""
Bandit recovery v4: task-aligned and more pessimistic wanting scenarios.

Key fixes relative to earlier versions:
  - Bounded beta: beta = beta_max * sigmoid(beta_x), no runaway beta.
  - Task outcome coding defaults to +10 / -10, not 0 / 1.
  - Probability schedule is explicit and configurable.
  - Two reversals default to trials 69 and 130.
  - Wanting scenarios include clean, noisy, low-variance, outcome-driven, mixed, and null.
  - Optional reward-history control models are fit alongside wanting models.

Run on Mac:
  python3 /Users/burgerks/Desktop/bandit_recovery_v4_task_aligned.py --n-agents 300 --seeds 20260615 20260616 20260617

Outputs default to /Users/burgerks/Desktop
"""

import argparse
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

N_ARMS = 3


def inv_logit(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=0)
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def softmax(beta, q):
    logits = beta * q
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / e.sum()


def make_schedule(n_trials, best_prob, mid_prob, worst_prob, reversals):
    # Initial arm mapping: A=best, B=worst, C=middle.
    cur = np.array([best_prob, worst_prob, mid_prob], dtype=float)
    out = []
    ri = 0
    for t in range(1, n_trials + 1):
        if ri < len(reversals) and t == reversals[ri]:
            # Reversal 1: A->B best. Reversal 2: B->C best.
            cur = cur[[2, 0, 1]] if ri % 2 == 0 else cur[[1, 2, 0]]
            ri += 1
        out.append(cur.copy())
    return np.asarray(out)


def make_rating_trials(n_trials):
    return np.array([1, 15, 29, 43, 57, 69, 82, 95, 108, 121, 130, 144, 158, 172, 186, 200], dtype=int)


def leaky_history_from_binary_rewards(binary_rewards, decay=0.75):
    """Causal leaky reward history using previous outcomes only."""
    h = np.zeros(len(binary_rewards), dtype=float)
    cur = 0.0
    for t in range(len(binary_rewards)):
        h[t] = cur
        cur = decay * cur + (1.0 - decay) * binary_rewards[t]
    return zscore(h)


def exogenous_ar_ratings(rng, n_ratings, ar=0.65):
    x = np.zeros(n_ratings, dtype=float)
    x[0] = rng.normal()
    for i in range(1, n_ratings):
        x[i] = ar * x[i - 1] + rng.normal(scale=np.sqrt(max(1e-8, 1.0 - ar * ar)))
    return zscore(x)


def carry_forward(rating_values, rating_trials, n_trials):
    trial = np.zeros(n_trials, dtype=float)
    idx = 0
    current = rating_values[0]
    for t in range(1, n_trials + 1):
        if idx < len(rating_trials) and t == rating_trials[idx]:
            current = rating_values[idx]
            idx += 1
        trial[t - 1] = current
    return trial


def simulate_agent(rng, cfg, scenario, true_alpha0, true_beta, true_phi):
    n_trials = cfg["n_trials"]
    schedule = cfg["schedule"]
    rating_trials = cfg["rating_trials"]
    n_ratings = len(rating_trials)

    # Pre-generate exogenous craving component at rating times.
    exog_rating = exogenous_ar_ratings(rng, n_ratings, ar=cfg["wanting_ar"])
    q = np.zeros(N_ARMS, dtype=float)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials, dtype=float)
    binary_rewards = np.zeros(n_trials, dtype=float)
    optimal = np.zeros(n_trials, dtype=bool)
    latent_want_trial = np.zeros(n_trials, dtype=float)
    observed_want_trial = np.zeros(n_trials, dtype=float)

    rating_idx = 0
    current_latent = exog_rating[0]
    current_observed = exog_rating[0]
    hist_cur = 0.0

    # Generate sequentially so outcome-driven wanting can depend on prior outcomes.
    for t in range(n_trials):
        trial_num = t + 1
        if rating_idx < n_ratings and trial_num == int(rating_trials[rating_idx]):
            exog_val = exog_rating[rating_idx]
            # Recent reward history available before this trial.
            hist_val = hist_cur
            if scenario == "clean_exogenous":
                latent = exog_val
                obs = latent
            elif scenario == "noisy_exogenous":
                latent = exog_val
                obs = cfg["rating_reliability"] * latent + np.sqrt(max(1e-8, 1 - cfg["rating_reliability"] ** 2)) * rng.normal()
            elif scenario == "low_variance_noisy":
                latent = exog_val
                obs = cfg["low_var_signal"] * latent + rng.normal(scale=cfg["low_var_noise"])
            elif scenario == "outcome_driven":
                latent = hist_val + rng.normal(scale=cfg["outcome_want_noise"])
                obs = latent
            elif scenario == "mixed":
                latent = cfg["mixed_exog_weight"] * exog_val + cfg["mixed_history_weight"] * hist_val + rng.normal(scale=cfg["mixed_noise"])
                obs = latent
            elif scenario == "noise_null":
                latent = exog_val
                obs = exog_val
            else:
                raise ValueError("Unknown scenario: {}".format(scenario))
            current_latent = latent
            current_observed = obs
            rating_idx += 1

        latent_want_trial[t] = current_latent
        observed_want_trial[t] = current_observed

        # Use latent wanting as the true state that modulates learning. Null has phi=0.
        a_t = inv_logit(logit(true_alpha0) + true_phi * current_latent)
        pc = softmax(true_beta, q)
        c = rng.choice(N_ARMS, p=pc)
        win = float(rng.random() < schedule[t, c])
        r = cfg["reward_value"] if win > 0 else cfg["loss_value"]
        q[c] += a_t * (r - q[c])
        choices[t] = c
        rewards[t] = r
        binary_rewards[t] = win
        optimal[t] = c == int(schedule[t].argmax())
        hist_cur = cfg["history_decay"] * hist_cur + (1.0 - cfg["history_decay"]) * win

    return {
        "choices": choices,
        "rewards": rewards,
        "binary_rewards": binary_rewards,
        "optimal": optimal,
        "latent_want_z": zscore(latent_want_trial),
        "observed_want_z": zscore(observed_want_trial),
        "reward_history_z": leaky_history_from_binary_rewards(binary_rewards, decay=cfg["history_decay"]),
    }


def unpack_params(x, model, beta_max):
    alpha0 = inv_logit(x[0])
    beta = beta_max * inv_logit(x[1])
    phi_want = 0.0
    phi_hist = 0.0
    if model == "wanting":
        phi_want = x[2]
    elif model == "history":
        phi_hist = x[2]
    elif model == "wanting_history":
        phi_want = x[2]
        phi_hist = x[3]
    elif model != "core":
        raise ValueError("Unknown model: {}".format(model))
    return alpha0, beta, phi_want, phi_hist


def neg_ll(x, model, choices, rewards, want_z, hist_z, beta_max):
    alpha0, beta, phi_want, phi_hist = unpack_params(x, model, beta_max)
    mod = phi_want * want_z + phi_hist * hist_z
    alpha_t = inv_logit(logit(alpha0) + mod)
    q = np.zeros(N_ARMS, dtype=float)
    ll = 0.0
    for t, c in enumerate(choices):
        p = softmax(beta, q)
        ll += np.log(max(p[c], 1e-300))
        q[c] += alpha_t[t] * (rewards[t] - q[c])
    return -ll


def fit_model(model, choices, rewards, want_z, hist_z, cfg):
    beta_max = cfg["beta_max"]
    # Keep this deliberately modest so full simulations are feasible.
    # The bounded parameterization removes the beta-runaway issue; a few starts are enough for a stable screen.
    beta_mid = min(max(np.median([cfg["true_beta_min"], cfg["true_beta_max"]]), 1e-4), beta_max * 0.8)
    base_starts = [
        (0.20, beta_mid),
        (0.50, beta_mid),
        (0.50, min(beta_max * 0.50, max(beta_mid * 2.0, 1e-4))),
    ]
    phi_starts = [0.0]
    if cfg.get("robust_starts", False):
        phi_starts = [-1.0, 0.0, 1.0]
        base_starts = [(0.15, beta_mid), (0.40, beta_mid), (0.70, beta_mid), (0.40, min(beta_max * 0.50, beta_mid * 2.0))]

    starts = []
    for a, b in base_starts:
        bx = logit(np.clip(b / beta_max, 1e-6, 1 - 1e-6))
        if model == "core":
            starts.append([logit(a), bx])
        elif model in ["wanting", "history"]:
            for p in phi_starts:
                starts.append([logit(a), bx, p])
        else:
            for pw in phi_starts:
                for ph in phi_starts:
                    starts.append([logit(a), bx, pw, ph])

    if model == "core":
        bounds = [(-6, 6), (-12, 12)]
    elif model in ["wanting", "history"]:
        bounds = [(-6, 6), (-12, 12), (-cfg["phi_bound"], cfg["phi_bound"])]
    else:
        bounds = [(-6, 6), (-12, 12), (-cfg["phi_bound"], cfg["phi_bound"]), (-cfg["phi_bound"], cfg["phi_bound"])]

    best_res = None
    best_fun = np.inf
    for s in starts:
        res = minimize(neg_ll, np.asarray(s, dtype=float), args=(model, choices, rewards, want_z, hist_z, beta_max), method="L-BFGS-B", bounds=bounds, options={"maxiter": cfg["maxiter"], "ftol": 1e-8})
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_res = res
    alpha0, beta, phi_want, phi_hist = unpack_params(best_res.x, model, beta_max)
    return alpha0, beta, phi_want, phi_hist, -best_fun


def corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 3 or x.std(ddof=0) < 1e-10 or y.std(ddof=0) < 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def run_seed(seed, n_agents, cfg):
    rng = np.random.default_rng(seed)
    rows = []
    scenarios = cfg["scenarios"]
    for scenario in scenarios:
        for agent in range(n_agents):
            true_alpha0 = rng.uniform(cfg["true_alpha_min"], cfg["true_alpha_max"])
            true_beta = np.exp(rng.uniform(np.log(cfg["true_beta_min"]), np.log(cfg["true_beta_max"])))
            true_phi = 0.0 if scenario == "noise_null" else rng.uniform(-cfg["true_phi_abs_max"], cfg["true_phi_abs_max"])
            sim = simulate_agent(rng, cfg, scenario, true_alpha0, true_beta, true_phi)
            choices = sim["choices"]; rewards = sim["rewards"]
            want_z = sim["observed_want_z"]; hist_z = sim["reward_history_z"]

            fits = {}
            for model in cfg["fit_models"]:
                fits[model] = fit_model(model, choices, rewards, want_z, hist_z, cfg)

            core = fits.get("core", (np.nan, np.nan, np.nan, np.nan, np.nan))
            want = fits.get("wanting", (np.nan, np.nan, np.nan, np.nan, np.nan))
            hist = fits.get("history", (np.nan, np.nan, np.nan, np.nan, np.nan))
            both = fits.get("wanting_history", (np.nan, np.nan, np.nan, np.nan, np.nan))

            rows.append({
                "seed": seed, "scenario": scenario, "agent": agent,
                "design": "{:.0f}/{:.0f}/{:.0f}, reversals {}, {} ratings, outcomes {}/{}".format(cfg["best_prob"]*100, cfg["mid_prob"]*100, cfg["worst_prob"]*100, cfg["reversals"], len(cfg["rating_trials"]), cfg["reward_value"], cfg["loss_value"]),
                "true_alpha0": true_alpha0, "true_beta": true_beta, "true_phi": true_phi,
                "core_alpha0": core[0], "core_beta": core[1], "core_ll": core[4],
                "want_alpha0": want[0], "want_beta": want[1], "want_phi": want[2], "want_ll": want[4],
                "hist_alpha0": hist[0], "hist_beta": hist[1], "hist_phi": hist[3], "hist_ll": hist[4],
                "both_alpha0": both[0], "both_beta": both[1], "both_phi_want": both[2], "both_phi_hist": both[3], "both_ll": both[4],
                "delta_ll_want_core": want[4] - core[4],
                "delta_ll_hist_core": hist[4] - core[4],
                "delta_ll_both_core": both[4] - core[4],
                "delta_ll_both_hist": both[4] - hist[4],
                "accuracy": float(sim["optimal"].mean()),
                "final40": float(sim["optimal"][-40:].mean()),
                "reward_rate": float(sim["binary_rewards"].mean()),
                "observed_want_sd_raw_proxy": float(np.std(want_z, ddof=0)),
                "want_history_corr": corr(want_z, hist_z),
            })
    return pd.DataFrame(rows)


def summarize_group(g):
    out = {
        "n": len(g),
        "core_alpha_r": corr(g.true_alpha0, g.core_alpha0),
        "core_alpha_mae": float(np.mean(np.abs(g.true_alpha0 - g.core_alpha0))),
        "core_beta_r": corr(np.log(g.true_beta), np.log(g.core_beta)),
        "core_log_beta_mae": float(np.mean(np.abs(np.log(g.true_beta) - np.log(g.core_beta)))),
        "want_alpha_r": corr(g.true_alpha0, g.want_alpha0),
        "want_beta_r": corr(np.log(g.true_beta), np.log(g.want_beta)),
        "want_phi_r": corr(g.true_phi, g.want_phi),
        "want_phi_s": spearman(g.true_phi, g.want_phi) if np.std(g.true_phi) > 0 else np.nan,
        "want_phi_mae": float(np.mean(np.abs(g.true_phi - g.want_phi))),
        "want_phi_sign_accuracy": float(np.mean(np.sign(g.true_phi) == np.sign(g.want_phi))) if np.std(g.true_phi) > 0 else np.nan,
        "both_phi_want_r": corr(g.true_phi, g.both_phi_want),
        "both_phi_want_s": spearman(g.true_phi, g.both_phi_want) if np.std(g.true_phi) > 0 else np.nan,
        "both_phi_want_sign_accuracy": float(np.mean(np.sign(g.true_phi) == np.sign(g.both_phi_want))) if np.std(g.true_phi) > 0 else np.nan,
        "mean_delta_ll_want_core": float(g.delta_ll_want_core.mean()),
        "pct_want_delta_ll_gt_2": float(np.mean(g.delta_ll_want_core > 2)),
        "pct_want_delta_ll_gt_4": float(np.mean(g.delta_ll_want_core > 4)),
        "mean_delta_ll_both_hist": float(g.delta_ll_both_hist.mean()),
        "pct_both_hist_delta_ll_gt_2": float(np.mean(g.delta_ll_both_hist > 2)),
        "pct_both_hist_delta_ll_gt_4": float(np.mean(g.delta_ll_both_hist > 4)),
        "mean_accuracy": float(g.accuracy.mean()),
        "pct_above_chance": float(np.mean(g.accuracy > (1.0/3.0))),
        "final40": float(g.final40.mean()),
        "reward_rate": float(g.reward_rate.mean()),
        "mean_want_history_corr": float(g.want_history_corr.mean()),
        "pct_beta_at_cap_core": float(np.mean(g.core_beta > 0.98 * g.core_beta.max())) if "core_beta" in g else np.nan,
    }
    return out


def make_summaries(raw):
    rows = []
    for scenario, g in raw.groupby("scenario", sort=False):
        row = summarize_group(g); row.update({"scenario": scenario, "seed": "pooled"}); rows.append(row)
    for (scenario, seed), g in raw.groupby(["scenario", "seed"], sort=False):
        row = summarize_group(g); row.update({"scenario": scenario, "seed": seed}); rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=50, help="Agents per seed per scenario. Use 300+ for validation.")
    p.add_argument("--seeds", type=int, nargs="+", default=[20260615, 20260616, 20260617])
    p.add_argument("--output-dir", type=str, default="/Users/burgerks/Desktop")
    p.add_argument("--prefix", type=str, default="bandit_recovery_v4")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--best-prob", type=float, default=0.80)
    p.add_argument("--mid-prob", type=float, default=0.50)
    p.add_argument("--worst-prob", type=float, default=0.25, help="Use 0.20 if your actual task worst arm is 20%.")
    p.add_argument("--reward-value", type=float, default=10.0)
    p.add_argument("--loss-value", type=float, default=-10.0)
    p.add_argument("--beta-max", type=float, default=2.0, help="Bound for beta under +/-10 coding. Use 10 only if you really want a very high cap.")
    p.add_argument("--true-beta-min", type=float, default=0.03)
    p.add_argument("--true-beta-max", type=float, default=0.80)
    p.add_argument("--true-alpha-min", type=float, default=0.08)
    p.add_argument("--true-alpha-max", type=float, default=0.85)
    p.add_argument("--true-phi-abs-max", type=float, default=1.5)
    p.add_argument("--phi-bound", type=float, default=3.0)
    p.add_argument("--history-decay", type=float, default=0.75)
    p.add_argument("--wanting-ar", type=float, default=0.65)
    p.add_argument("--rating-reliability", type=float, default=0.65)
    p.add_argument("--low-var-signal", type=float, default=0.25)
    p.add_argument("--low-var-noise", type=float, default=1.00)
    p.add_argument("--outcome-want-noise", type=float, default=0.50)
    p.add_argument("--mixed-exog-weight", type=float, default=0.60)
    p.add_argument("--mixed-history-weight", type=float, default=0.60)
    p.add_argument("--mixed-noise", type=float, default=0.50)
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--robust-starts", action="store_true", help="Use more optimizer starts; slower but more stable.")
    p.add_argument("--scenarios", nargs="+", default=["clean_exogenous", "noisy_exogenous", "low_variance_noisy", "outcome_driven", "mixed", "noise_null"])
    p.add_argument("--fit-models", nargs="+", default=["core", "wanting", "history", "wanting_history"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args).copy()
    cfg["reversals"] = [69, 130]
    cfg["rating_trials"] = make_rating_trials(args.n_trials)
    cfg["schedule"] = make_schedule(args.n_trials, args.best_prob, args.mid_prob, args.worst_prob, cfg["reversals"])

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("Output directory: {}".format(out_dir))
    print("Design: best/mid/worst = {}/{}/{}; outcomes = {}/{}; beta_max = {}".format(args.best_prob, args.mid_prob, args.worst_prob, args.reward_value, args.loss_value, args.beta_max))
    print("Reversals: {}; rating trials: {}".format(cfg["reversals"], list(cfg["rating_trials"])))
    print("Seeds: {}; agents per seed per scenario: {}".format(args.seeds, args.n_agents))

    parts = []
    for seed in args.seeds:
        print("Running seed {}...".format(seed))
        parts.append(run_seed(seed, args.n_agents, cfg))
    raw = pd.concat(parts, ignore_index=True)
    summary = make_summaries(raw)

    raw_path = out_dir / "{}_raw.csv".format(args.prefix)
    summary_path = out_dir / "{}_summary.csv".format(args.prefix)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    keep = ["scenario", "seed", "n", "core_alpha_r", "core_beta_r", "want_phi_r", "want_phi_s", "want_phi_sign_accuracy", "both_phi_want_r", "both_phi_want_sign_accuracy", "pct_want_delta_ll_gt_2", "pct_want_delta_ll_gt_4", "pct_both_hist_delta_ll_gt_2", "mean_accuracy", "pct_above_chance", "mean_want_history_corr"]
    print("\nSUMMARY")
    print(summary[keep].to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nSaved:\n  {}\n  {}".format(raw_path, summary_path))
    print("Elapsed minutes: {:.2f}".format((time.time() - start) / 60.0))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
