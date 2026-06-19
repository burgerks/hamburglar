#!/usr/bin/env python3
"""
Hierarchical recovery for residual food-cued vigor predicting bandit learning.

Design decision carried in from the earlier confound analysis: rho (reward
sensitivity) and beta (inverse temperature) are not separately identifiable with a
single reward magnitude, so rho is FIXED to 1 and beta is read as value sensitivity.

Model (per subject i), rho = 1:
    choice_t ~ softmax(beta_i * Q_t)
    pe_t = r_t - Q_t[chosen];  a = alpha_pos_i if pe>=0 else alpha_neg_i
    Q[chosen] += a * pe_t

Group level: residual food vigor enters as a regressor on every fitted parameter,
on its natural (unbounded) scale:
    logit(alpha_pos_i) = m_ap + gamma_ap * vigor_i + e
    logit(alpha_neg_i) = m_an + gamma_an * vigor_i + e
    log(beta_i)        = m_b  + gamma_b  * vigor_i + e
The gamma_* are the brain-behavior effects of interest.

Fitting: hierarchical EM over a parameter grid (partial pooling). Each EM iteration
computes the per-subject posterior over the grid given the current group prior,
takes posterior means/variances (E-step), then updates the group means, the gamma
regression coefficients, and the between-subject variances (M-step). This pools
information across subjects and de-attenuates the gamma estimates relative to the
fit-each-subject-then-correlate approach.

Scenarios generate a true vigor effect on one fitted parameter (alpha_pos,
alpha_neg, or beta) or none (null). The former reward-sensitivity hypothesis is
tested as the beta effect, since with fixed magnitude rho is absorbed into beta.

Outputs:
    summary_recovery_hier.csv     one row per scenario (the documented variables)
    dataset_recovery_hier.csv     one row per simulated dataset
"""

import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inlined generative model (self-contained; mirrors the task and the earlier
# food-residual-vigor simulation, so no external module import is needed).
# ---------------------------------------------------------------------------
N_TRIALS = 200
N_ARMS = 3
REVERSAL_TRIALS = [69, 130]                  # 1-indexed; reversal takes effect on that trial
PROFILES0 = np.array([[0.80, 0.20], [0.30, 0.70], [0.50, 0.50]], dtype=float)
WIN_REWARD, LOSS_REWARD = 1.0, -1.0
N_FOOD_MID, N_NEUTRAL_MID = 16, 14


def softmax_probs(q, beta):
    z = beta * q; z -= np.max(z); e = np.exp(z); return e / np.sum(e)


def ols_slope_intercept(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1]), float(coef[0])


def zscore_(x):
    x = np.asarray(x, float); return (x - np.mean(x)) / (np.std(x, ddof=0) + 1e-12)


def simulate_mid_rts(rng, n_subjects):
    """Simulate Mini-MID RTs and compute residual food vigor per subject:
    residual_food_vigor = -zscore(residual of median food RT on median neutral RT)."""
    rows = []
    for sid in range(n_subjects):
        gs = rng.normal(0, 1)                    # general MID speed (shared)
        fv = rng.normal(0, 1)                    # food-specific vigor
        neutral = np.clip(350 - 38 * gs + rng.normal(0, 35, N_NEUTRAL_MID), 180, 700)
        food = np.clip(350 - 38 * gs - 32 * fv + rng.normal(0, 35, N_FOOD_MID), 180, 700)
        rows.append({"subject": sid,
                     "median_food_mid_rt": float(np.median(food)),
                     "median_neutral_mid_rt": float(np.median(neutral))})
    df = pd.DataFrame(rows)
    slope, intercept = ols_slope_intercept(df["median_neutral_mid_rt"], df["median_food_mid_rt"])
    resid = df["median_food_mid_rt"].to_numpy() - (intercept + slope * df["median_neutral_mid_rt"].to_numpy())
    df["residual_food_vigor"] = -zscore_(resid)
    return df


def generate_reward_matrix(rng):
    """Per-trial +1/-1 outcomes for each arm, with random-direction reversals."""
    profiles = PROFILES0.copy()
    rmat = np.zeros((N_TRIALS, N_ARMS))
    for t in range(1, N_TRIALS + 1):
        if t in REVERSAL_TRIALS:
            profiles = profiles[[2, 0, 1], :] if rng.random() < 0.5 else profiles[[1, 2, 0], :]
        for arm in range(N_ARMS):
            rmat[t - 1, arm] = WIN_REWARD if rng.random() < profiles[arm, 0] else LOSS_REWARD
    return rmat


def simulate_choices(rng, params, reward_matrix):
    """One agent's choices under the asymmetric-LR, reward-sensitivity model."""
    ap, an, rho, beta = params["alpha_pos"], params["alpha_neg"], params["rho"], params["beta"]
    q = np.zeros(N_ARMS); choices = np.zeros(N_TRIALS, int); rewards = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        p = softmax_probs(q, beta)
        c = int(rng.choice(N_ARMS, p=p)); r = float(reward_matrix[t, c])
        pe = rho * r - q[c]
        q[c] += (ap if pe >= 0 else an) * pe
        choices[t] = c; rewards[t] = r
    return choices, rewards, None


SCENARIOS = ["alpha_pos", "alpha_neg", "beta", "null"]
# Which fitted parameter each scenario truly acts on (for the "largest gamma" check).
TARGET = {"alpha_pos": "alpha_pos", "alpha_neg": "alpha_neg", "beta": "beta", "null": None}
PARAMS = ["alpha_pos", "alpha_neg", "beta"]


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
def logit(p): return np.log(p / (1 - p))


def make_params(rng, vigor, scenario, gamma):
    """True subject parameters with rho fixed to 1; gamma*vigor added to one param."""
    ap = rng.normal(-1.05, 0.55)
    an = rng.normal(-1.35, 0.55)
    lb = rng.normal(np.log(3.0), 0.45)
    if scenario == "alpha_pos": ap += gamma * vigor
    elif scenario == "alpha_neg": an += gamma * vigor
    elif scenario == "beta": lb += gamma * vigor
    return dict(alpha_pos=float(sigmoid(ap)), alpha_neg=float(sigmoid(an)),
                rho=1.0, beta=float(np.clip(np.exp(lb), 0.2, 25.0)),
                ap_logit=float(ap), an_logit=float(an), log_beta=float(lb))


# ----- parameter grid (transformed scale) -----
def build_grid(n_ap=11, n_an=11, n_b=11):
    ap = np.linspace(logit(0.04), logit(0.93), n_ap)
    an = np.linspace(logit(0.04), logit(0.93), n_an)
    lb = np.linspace(np.log(0.3), np.log(20.0), n_b)
    A, B, C = np.meshgrid(ap, an, lb, indexing="ij")
    grid = np.column_stack([A.ravel(), B.ravel(), C.ravel()])   # (G,3): ap_logit, an_logit, log_beta
    nat = np.column_stack([sigmoid(grid[:, 0]), sigmoid(grid[:, 1]), np.exp(grid[:, 2])])
    return grid, nat


def data_nll(nat_ap, nat_an, nat_beta, ch, rw):
    """Vectorized data NLL over all grid cells for one subject (rho = 1)."""
    G = nat_beta.size; q = np.zeros((G, 3)); nll = np.zeros(G); idx = np.arange(G)
    for c, r in zip(ch, rw):
        z = nat_beta[:, None] * q; z -= z.max(1, keepdims=True)
        e = np.exp(z); p = e / e.sum(1, keepdims=True)
        nll -= np.log(p[idx, c] + 1e-12)
        pe = r - q[idx, c]
        a = np.where(pe >= 0, nat_ap, nat_an)
        q[idx, c] += a * pe
    return nll


def em_fit(ND, grid, vigor, n_iter=50):
    """Hierarchical EM over the grid. ND is (S,G) data NLL. Returns group means m,
    gamma coefficients (per param), between-subject var Sigma, and per-subject
    posterior means Etheta (S,3) on the transformed scale."""
    S, G = ND.shape
    m = np.array([-1.05, -1.35, np.log(3.0)])
    gamma = np.zeros(3)
    Sigma = np.array([0.5, 0.5, 0.4]) ** 2
    vz = (vigor - vigor.mean()) / (vigor.std() + 1e-12)
    sx2 = float(np.sum((vz - vz.mean()) ** 2))
    Etheta = np.zeros((S, 3))
    for _ in range(n_iter):
        mu = m[None, :] + np.outer(vz, gamma)            # (S,3) prior means
        pen = np.zeros((S, G))
        for d in range(3):
            diff = grid[:, d][None, :] - mu[:, d][:, None]
            pen += 0.5 * diff ** 2 / Sigma[d]
        logw = -(ND + pen)
        logw -= logw.max(1, keepdims=True)
        w = np.exp(logw); w /= w.sum(1, keepdims=True)   # (S,G) posterior over cells
        Etheta = w @ grid                                # (S,3)
        Etheta2 = w @ (grid ** 2)
        Vtheta = np.clip(Etheta2 - Etheta ** 2, 0, None)
        # M-step: regress posterior means on vigor; update means, gammas, variances.
        for d in range(3):
            b = float(np.sum((vz - vz.mean()) * (Etheta[:, d] - Etheta[:, d].mean())) / (sx2 + 1e-12))
            a0 = float(Etheta[:, d].mean() - b * vz.mean())
            gamma[d] = b; m[d] = a0
            resid = Etheta[:, d] - (a0 + b * vz)
            Sigma[d] = float(np.mean(resid ** 2 + Vtheta[:, d]))
    se = np.sqrt(Sigma / (sx2 + 1e-12))                  # Wald SE for each gamma
    return m, gamma, se, Sigma, Etheta


def run(args):
    rng = np.random.default_rng(args.seed)
    grid, nat = build_grid(args.grid, args.grid, args.grid)
    nat_ap, nat_an, nat_b = nat[:, 0], nat[:, 1], nat[:, 2]
    Z = 1.96

    ds_rows = []
    for sc in SCENARIOS:
        for d in range(args.n_datasets):
            mid = simulate_mid_rts(rng, args.n_subjects)
            vig = mid["residual_food_vigor"].to_numpy()
            S = len(vig)
            ND = np.zeros((S, nat_b.size))
            true_t = np.zeros((S, 3)); true_nat = np.zeros((S, 3))
            for i, v in enumerate(vig):
                tp = make_params(rng, float(v), sc, args.gamma)
                rmat = generate_reward_matrix(rng)
                ch, rw, _ = simulate_choices(rng, tp, rmat)
                ND[i] = data_nll(nat_ap, nat_an, nat_b, ch, rw)
                true_t[i] = [tp["ap_logit"], tp["an_logit"], tp["log_beta"]]
                true_nat[i] = [tp["alpha_pos"], tp["alpha_neg"], tp["beta"]]
            m, gamma, se, Sigma, Eth = em_fit(ND, grid, vig, n_iter=args.em_iter)
            est_nat = np.column_stack([sigmoid(Eth[:, 0]), sigmoid(Eth[:, 1]), np.exp(Eth[:, 2])])

            row = {"scenario": sc, "dataset": d}
            for j, p in enumerate(PARAMS):
                row[f"recov_r_{p}"] = float(np.corrcoef(true_nat[:, j], est_nat[:, j])[0, 1])
                row[f"est_gamma_{p}"] = float(gamma[j])
                row[f"se_gamma_{p}"] = float(se[j])
                lo, hi = gamma[j] - Z * se[j], gamma[j] + Z * se[j]
                row[f"ci_excludes0_{p}"] = int(lo > 0 or hi < 0)
                tg = args.gamma if TARGET[sc] == p else 0.0
                row[f"true_gamma_{p}"] = tg
                row[f"ci_covers_true_{p}"] = int(lo <= tg <= hi)
            if TARGET[sc] is not None:
                row["largest_gamma_param"] = PARAMS[int(np.argmax(np.abs(gamma)))]
                row["target_is_largest"] = int(row["largest_gamma_param"] == TARGET[sc])
            ds_rows.append(row)
        print(f"  [{sc}] {args.n_datasets} datasets done", flush=True)

    ds = pd.DataFrame(ds_rows)

    # ---- per-scenario summary (the documented output variables) ----
    sum_rows = []
    for sc, g in ds.groupby("scenario"):
        r = {"scenario": sc, "n_datasets": int(len(g)), "n_subjects": args.n_subjects,
             "true_gamma_setting": (0.0 if sc == "null" else args.gamma)}
        for p in PARAMS:
            r[f"mean_recov_r_{p}"] = float(g[f"recov_r_{p}"].mean())
            r[f"sd_recov_r_{p}"] = float(g[f"recov_r_{p}"].std(ddof=1)) if len(g) > 1 else np.nan
            r[f"mean_est_gamma_{p}"] = float(g[f"est_gamma_{p}"].mean())
            r[f"sd_est_gamma_{p}"] = float(g[f"est_gamma_{p}"].std(ddof=1)) if len(g) > 1 else np.nan
            r[f"true_gamma_{p}"] = float(g[f"true_gamma_{p}"].iloc[0])
            r[f"gamma_bias_{p}"] = r[f"mean_est_gamma_{p}"] - r[f"true_gamma_{p}"]
            r[f"detect_rate_gamma_{p}"] = float(g[f"ci_excludes0_{p}"].mean())   # power if target, FPR if not
            r[f"ci_coverage_gamma_{p}"] = float(g[f"ci_covers_true_{p}"].mean())
        if sc != "null":
            r["target_recovered_as_largest_gamma_rate"] = float(g["target_is_largest"].mean())
        sum_rows.append(r)
    summary = pd.DataFrame(sum_rows)

    ds.to_csv(args.outdir + "/dataset_recovery_hier.csv", index=False)
    summary.to_csv(args.outdir + "/summary_recovery_hier.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 100)
    print("\nSUMMARY (hierarchical, rho fixed):")
    cols = ["scenario", "n_datasets", "mean_recov_r_alpha_pos", "mean_recov_r_alpha_neg",
            "mean_recov_r_beta", "mean_est_gamma_alpha_pos", "mean_est_gamma_alpha_neg",
            "mean_est_gamma_beta", "detect_rate_gamma_alpha_pos", "detect_rate_gamma_alpha_neg",
            "detect_rate_gamma_beta"]
    print(summary[cols].to_string(index=False))
    return summary


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n-datasets", type=int, default=300, dest="n_datasets")
    p.add_argument("--n-subjects", type=int, default=75, dest="n_subjects")
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--grid", type=int, default=11, help="grid points per parameter dimension")
    p.add_argument("--em-iter", type=int, default=50, dest="em_iter")
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
