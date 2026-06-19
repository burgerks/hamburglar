"""
Single-subject simulate-and-recover for selected probabilistic 3-arm bandit.

Design:
  - 200 trials
  - reward probabilities 80/50/25
  - reversals before trials 69 and 130
  - 16 wanting ratings

Models fit to each simulated agent:
  1) core: alpha + beta
  2) phi:  alpha0 + beta + phi, where
        alpha_t = sigmoid(logit(alpha0) + phi * wanting_z_t)

Simulation scenarios:
  1) signal_phi: true phi varies across agents
  2) noise_null: wanting is measured but has no true effect on learning; true phi = 0

Outputs are saved by default to the user's Desktop.
"""

import argparse
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

N_TRIALS = 200
N_ARMS = 3
PROBS = np.array([0.80, 0.25, 0.50], dtype=float)  # A best, B worst, C middle at start
REVERSALS = [69, 130]
WANTING_BEFORE_TRIALS = np.array([1, 15, 29, 43, 57, 69, 82, 95, 108, 121, 130, 144, 158, 172, 186, 200])

# Broader/finer starting grids. Continuous optimization is then run from the best grid point.
ALPHA_GRID_CORE = np.linspace(0.03, 0.95, 21)
BETA_GRID_CORE = np.exp(np.linspace(np.log(0.25), np.log(15.0), 23))

ALPHA_GRID_PHI = np.linspace(0.03, 0.95, 17)
BETA_GRID_PHI = np.exp(np.linspace(np.log(0.25), np.log(15.0), 19))
PHI_GRID = np.linspace(-3.0, 3.0, 21)


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def softmax_probs(beta, q):
    logits = beta * q
    logits = logits - np.max(logits)
    p = np.exp(logits)
    return p / p.sum()


def make_schedule():
    """Return trial x arm reward probabilities with deterministic reversals."""
    cur = PROBS.copy()
    out = []
    ri = 0
    for t in range(1, N_TRIALS + 1):
        if ri < len(REVERSALS) and t == REVERSALS[ri]:
            # first reversal rotates [C,A,B]; second reversal rotates [B,C,A]
            cur = cur[[2, 0, 1]] if ri % 2 == 0 else cur[[1, 2, 0]]
            ri += 1
        out.append(cur.copy())
    return np.asarray(out)


SCHEDULE = make_schedule()


def generate_wanting_trace(rng, ar=0.65):
    """Generate 16 observed wanting ratings and carry the latest rating forward to each trial."""
    z = np.zeros(len(WANTING_BEFORE_TRIALS), dtype=float)
    z[0] = rng.normal()
    for i in range(1, len(z)):
        z[i] = ar * z[i - 1] + rng.normal(scale=np.sqrt(max(1e-8, 1 - ar**2)))
    z = (z - z.mean()) / (z.std(ddof=0) + 1e-8)

    trial_z = np.zeros(N_TRIALS, dtype=float)
    rating_idx = 0
    current = z[0]
    for t in range(1, N_TRIALS + 1):
        if rating_idx < len(WANTING_BEFORE_TRIALS) and t == WANTING_BEFORE_TRIALS[rating_idx]:
            current = z[rating_idx]
            rating_idx += 1
        trial_z[t - 1] = current
    return trial_z, z


def simulate_agent(rng, alpha0, beta, phi, wanting_z):
    q = np.zeros(N_ARMS, dtype=float)
    choices = np.zeros(N_TRIALS, dtype=int)
    rewards = np.zeros(N_TRIALS, dtype=float)
    optimal = np.zeros(N_TRIALS, dtype=bool)
    alpha_t = inv_logit(logit(alpha0) + phi * wanting_z)

    for t in range(N_TRIALS):
        p_choice = softmax_probs(beta, q)
        c = rng.choice(N_ARMS, p=p_choice)
        r = float(rng.random() < SCHEDULE[t, c])
        q[c] += alpha_t[t] * (r - q[c])
        choices[t] = c
        rewards[t] = r
        optimal[t] = c == int(SCHEDULE[t].argmax())
    return choices, rewards, optimal


def neg_ll_core_unconstrained(x, choices, rewards):
    # x[0] = logit alpha, x[1] = log beta
    alpha = inv_logit(x[0])
    beta = np.exp(x[1])
    q = np.zeros(N_ARMS, dtype=float)
    ll = 0.0
    for c, r in zip(choices, rewards):
        logits = beta * q
        mx = np.max(logits)
        ll += logits[c] - (mx + np.log(np.exp(logits - mx).sum()))
        q[c] += alpha * (r - q[c])
    return -ll


def neg_ll_phi_unconstrained(x, choices, rewards, wanting_z):
    # x[0] = logit alpha0, x[1] = log beta, x[2] = phi
    alpha0 = inv_logit(x[0])
    beta = np.exp(x[1])
    phi = x[2]
    alpha_t = inv_logit(logit(alpha0) + phi * wanting_z)
    q = np.zeros(N_ARMS, dtype=float)
    ll = 0.0
    for t, (c, r) in enumerate(zip(choices, rewards)):
        logits = beta * q
        mx = np.max(logits)
        ll += logits[c] - (mx + np.log(np.exp(logits - mx).sum()))
        q[c] += alpha_t[t] * (r - q[c])
    return -ll


def coarse_fit_core(choices, rewards):
    best_nll = np.inf
    best = (0.3, 2.0)
    for alpha in ALPHA_GRID_CORE:
        for beta in BETA_GRID_CORE:
            nll = neg_ll_core_unconstrained([logit(alpha), np.log(beta)], choices, rewards)
            if nll < best_nll:
                best_nll = nll
                best = (alpha, beta)
    return best[0], best[1], best_nll


def coarse_fit_phi(choices, rewards, wanting_z):
    best_nll = np.inf
    best = (0.3, 2.0, 0.0)
    for alpha0 in ALPHA_GRID_PHI:
        la = logit(alpha0)
        for beta in BETA_GRID_PHI:
            lb = np.log(beta)
            for phi in PHI_GRID:
                nll = neg_ll_phi_unconstrained([la, lb, phi], choices, rewards, wanting_z)
                if nll < best_nll:
                    best_nll = nll
                    best = (alpha0, beta, phi)
    return best[0], best[1], best[2], best_nll


def fit_core(choices, rewards, use_optimizer=True):
    alpha_start, beta_start, nll_start = coarse_fit_core(choices, rewards)
    if use_optimizer and SCIPY_AVAILABLE:
        res = minimize(
            neg_ll_core_unconstrained,
            x0=np.array([logit(alpha_start), np.log(beta_start)]),
            args=(choices, rewards),
            method="Nelder-Mead",
            options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-6},
        )
        if res.success or res.fun < nll_start:
            return float(inv_logit(res.x[0])), float(np.exp(res.x[1])), float(-res.fun)
    return float(alpha_start), float(beta_start), float(-nll_start)


def fit_phi(choices, rewards, wanting_z, use_optimizer=True):
    alpha_start, beta_start, phi_start, nll_start = coarse_fit_phi(choices, rewards, wanting_z)
    if use_optimizer and SCIPY_AVAILABLE:
        res = minimize(
            neg_ll_phi_unconstrained,
            x0=np.array([logit(alpha_start), np.log(beta_start), phi_start]),
            args=(choices, rewards, wanting_z),
            method="Nelder-Mead",
            options={"maxiter": 2500, "xatol": 1e-6, "fatol": 1e-6},
        )
        if res.success or res.fun < nll_start:
            return float(inv_logit(res.x[0])), float(np.exp(res.x[1])), float(res.x[2]), float(-res.fun)
    return float(alpha_start), float(beta_start), float(phi_start), float(-nll_start)


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def run_seed(seed, n_agents, use_optimizer=True):
    rng = np.random.default_rng(seed)
    rows = []
    scenarios = ["signal_phi", "noise_null"]

    for scenario in scenarios:
        for agent in range(n_agents):
            true_alpha0 = rng.uniform(0.08, 0.85)
            true_beta = np.exp(rng.uniform(np.log(0.50), np.log(8.0)))
            true_phi = rng.uniform(-1.50, 1.50) if scenario == "signal_phi" else 0.0
            wanting_trial_z, wanting_ratings_z = generate_wanting_trace(rng)
            choices, rewards, optimal = simulate_agent(rng, true_alpha0, true_beta, true_phi, wanting_trial_z)

            core_alpha, core_beta, core_ll = fit_core(choices, rewards, use_optimizer=use_optimizer)
            phi_alpha, phi_beta, rec_phi, phi_ll = fit_phi(choices, rewards, wanting_trial_z, use_optimizer=use_optimizer)

            rows.append({
                "seed": seed,
                "scenario": scenario,
                "agent": agent,
                "design": "80/50/25, 2 reversals @69/130, 16 wanting",
                "true_alpha0": true_alpha0,
                "true_beta": true_beta,
                "true_phi": true_phi,
                "core_rec_alpha0": core_alpha,
                "core_rec_beta": core_beta,
                "core_ll": core_ll,
                "phi_rec_alpha0": phi_alpha,
                "phi_rec_beta": phi_beta,
                "rec_phi": rec_phi,
                "phi_ll": phi_ll,
                "delta_ll_phi_minus_core": phi_ll - core_ll,
                "accuracy": float(optimal.mean()),
                "final40": float(optimal[-40:].mean()),
                "reward_rate": float(rewards.mean()),
                "mean_alpha_t": float(inv_logit(logit(true_alpha0) + true_phi * wanting_trial_z).mean()),
                "sd_alpha_t": float(inv_logit(logit(true_alpha0) + true_phi * wanting_trial_z).std(ddof=0)),
            })
    return pd.DataFrame(rows)


def summarize(g):
    return pd.Series({
        "n": len(g),
        "core_alpha_r": corr(g.true_alpha0, g.core_rec_alpha0),
        "core_alpha_mae": np.mean(np.abs(g.true_alpha0 - g.core_rec_alpha0)),
        "core_beta_r": corr(np.log(g.true_beta), np.log(g.core_rec_beta)),
        "core_log_beta_mae": np.mean(np.abs(np.log(g.true_beta) - np.log(g.core_rec_beta))),
        "phi_model_alpha_r": corr(g.true_alpha0, g.phi_rec_alpha0),
        "phi_model_alpha_mae": np.mean(np.abs(g.true_alpha0 - g.phi_rec_alpha0)),
        "phi_model_beta_r": corr(np.log(g.true_beta), np.log(g.phi_rec_beta)),
        "phi_model_log_beta_mae": np.mean(np.abs(np.log(g.true_beta) - np.log(g.phi_rec_beta))),
        "phi_r": corr(g.true_phi, g.rec_phi) if np.std(g.true_phi) > 0 else np.nan,
        "phi_s": spearman(g.true_phi, g.rec_phi) if np.std(g.true_phi) > 0 else np.nan,
        "phi_mae": np.mean(np.abs(g.true_phi - g.rec_phi)),
        "phi_sign_accuracy": np.mean(np.sign(g.true_phi) == np.sign(g.rec_phi)) if np.std(g.true_phi) > 0 else np.nan,
        "mean_delta_ll_phi_minus_core": g.delta_ll_phi_minus_core.mean(),
        "median_delta_ll_phi_minus_core": g.delta_ll_phi_minus_core.median(),
        "mean_accuracy": g.accuracy.mean(),
        "pct_above_chance": np.mean(g.accuracy > (1/3)),
        "final40": g.final40.mean(),
        "reward_rate": g.reward_rate.mean(),
        "mean_alpha_t_sd": g.sd_alpha_t.mean(),
    })


def model_comparison_summary(raw):
    rows = []
    for (scenario, seed), g in raw.groupby(["scenario", "seed"], sort=False):
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "n": len(g),
            "mean_delta_ll_phi_minus_core": g.delta_ll_phi_minus_core.mean(),
            "median_delta_ll_phi_minus_core": g.delta_ll_phi_minus_core.median(),
            "pct_phi_ll_better_than_core": np.mean(g.delta_ll_phi_minus_core > 0),
            "pct_delta_ll_gt_2": np.mean(g.delta_ll_phi_minus_core > 2),
            "pct_delta_ll_gt_4": np.mean(g.delta_ll_phi_minus_core > 4),
        })
    pooled = []
    for scenario, g in raw.groupby("scenario", sort=False):
        pooled.append({
            "scenario": scenario,
            "seed": "pooled",
            "n": len(g),
            "mean_delta_ll_phi_minus_core": g.delta_ll_phi_minus_core.mean(),
            "median_delta_ll_phi_minus_core": g.delta_ll_phi_minus_core.median(),
            "pct_phi_ll_better_than_core": np.mean(g.delta_ll_phi_minus_core > 0),
            "pct_delta_ll_gt_2": np.mean(g.delta_ll_phi_minus_core > 2),
            "pct_delta_ll_gt_4": np.mean(g.delta_ll_phi_minus_core > 4),
        })
    return pd.DataFrame(pooled + rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Bandit simulate-and-recover with optional wanting modulation.")
    parser.add_argument("--n-agents", type=int, default=100, help="Agents per seed per scenario. Use 300-500 for a heavier run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260615, 20260616, 20260617], help="Random seeds.")
    parser.add_argument("--output-dir", type=str, default="/Users/burgerks/Desktop", help="Folder where CSVs will be written.")
    parser.add_argument("--prefix", type=str, default="bandit_recovery_phi", help="Prefix for output CSV names.")
    parser.add_argument("--no-optimizer", action="store_true", help="Use coarse grid only; faster but less precise.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("Output directory: {}".format(out_dir))
    print("Seeds: {}".format(args.seeds))
    print("Agents per seed per scenario: {}".format(args.n_agents))
    print("Optimizer available: {}; using optimizer: {}".format(SCIPY_AVAILABLE, (not args.no_optimizer and SCIPY_AVAILABLE)))

    raw_parts = []
    for seed in args.seeds:
        print("Running seed {}...".format(seed))
        raw_parts.append(run_seed(seed, args.n_agents, use_optimizer=(not args.no_optimizer and SCIPY_AVAILABLE)))
    raw = pd.concat(raw_parts, ignore_index=True)

    # Avoid pandas groupby.apply deprecation warning by summarizing explicitly.
    summary_rows = []
    for scenario, g in raw.groupby("scenario", sort=False):
        row = summarize(g).to_dict()
        row.update({"scenario": scenario, "seed": "pooled"})
        summary_rows.append(row)
    for (scenario, seed), g in raw.groupby(["scenario", "seed"], sort=False):
        row = summarize(g).to_dict()
        row.update({"scenario": scenario, "seed": seed})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    model_comp = model_comparison_summary(raw)

    raw_path = out_dir / "{}_raw.csv".format(args.prefix)
    summary_path = out_dir / "{}_summary.csv".format(args.prefix)
    model_path = out_dir / "{}_model_comparison.csv".format(args.prefix)

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    model_comp.to_csv(model_path, index=False)

    cols = [
        "scenario", "seed", "n",
        "core_alpha_r", "core_alpha_mae", "core_beta_r", "core_log_beta_mae",
        "phi_model_alpha_r", "phi_model_alpha_mae", "phi_model_beta_r", "phi_model_log_beta_mae",
        "phi_r", "phi_mae", "phi_sign_accuracy",
        "mean_accuracy", "pct_above_chance", "final40", "reward_rate",
    ]
    print("\nSUMMARY")
    print(summary[cols].to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nMODEL COMPARISON")
    print(model_comp.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nSaved:\n  {}\n  {}\n  {}".format(raw_path, summary_path, model_path))
    print("Elapsed minutes: {:.2f}".format((time.time() - start) / 60))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
