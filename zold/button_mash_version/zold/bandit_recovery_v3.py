#!/usr/bin/env python3
"""
Realistic single-subject simulate-and-recover for the 3-arm bandit.

Corrections vs the previous version:
  - beta is BOUNDED (beta = BMAX / (1 + e^-x), BMAX = 10), so it cannot diverge.
  - symmetric win/loss coding (+1 / -1), matching the task's +10/-10 payoff
    rescaled (beta on the +/-1 scale is 1/10 of beta on the +/-10 scale).
  - exact task structure: reward probs 80/50/30, random-direction reversals at
    trials 69 and 130, 16 wanting ratings at the task's trials.

Wanting (craving) scenarios, to get a realistic floor on phi recovery:
  1) signal_clean   exogenous AR(0.65) craving, measured without error (optimistic).
  2) noisy_wanting  same craving drives learning, but the rating the model sees is
                    a noisy readout (corr(observed,true) ~ 0.65). Errors-in-variables.
  3) outcome_driven craving is a leaky integrator of recent rewards, sampled only at
                    the 16 rating trials. Endogenous craving plus intermittent sampling.
  4) noise_null     true phi = 0, clean craving. False-positive / calibration check.

Two models are grid-fit per agent: core (alpha, beta) and phi (alpha0, beta, phi),
where alpha_t = sigmoid(logit(alpha0) + phi * craving_z_t).
"""

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

# ----- task structure -----
N = 200
PROBS0 = np.array([0.80, 0.30, 0.50])           # best, worst, middle (task PROFILE_A/B/C)
REVERSALS = [69, 130]
RATING_TRIALS = np.array([1,15,29,43,57,69,82,95,108,121,130,144,158,172,186,200])
WIN, LOSS = 1.0, -1.0
BMAX = 10.0                                       # beta cap (Kulkarni-style bound)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))

# ----- recovery grids -----
A_GRID = np.linspace(0.03, 0.95, 13)
B_GRID = np.geomspace(0.3, 10.0, 13)
P_GRID = np.linspace(-3.0, 3.0, 17)
cA, cB = np.meshgrid(A_GRID, B_GRID, indexing="ij")
CORE_A, CORE_B = cA.ravel(), cB.ravel()
pA, pB, pP = np.meshgrid(A_GRID, B_GRID, P_GRID, indexing="ij")
PHI_A, PHI_B, PHI_P = pA.ravel(), pB.ravel(), pP.ravel()
PHI_LA = logit(PHI_A)

def make_schedule(rng):
    """Per-trial 3-arm reward probabilities with random-direction reversals (as in the task)."""
    cur = PROBS0.copy(); out = []; ri = 0
    for t in range(1, N+1):
        if ri < len(REVERSALS) and t == REVERSALS[ri]:
            cur = cur[[2,0,1]] if rng.random() < 0.5 else cur[[1,2,0]]
            ri += 1
        out.append(cur.copy())
    return np.asarray(out)

def ar_ratings(rng, ar=0.65):
    """16 z-scored craving ratings from an AR(0.65) latent."""
    k = len(RATING_TRIALS); z = np.zeros(k); z[0] = rng.normal()
    for i in range(1, k):
        z[i] = ar*z[i-1] + rng.normal(scale=np.sqrt(1-ar**2))
    return (z - z.mean())/(z.std()+1e-8)

def forward_fill(ratings):
    """Carry the most recent rating forward to every trial (what the model uses)."""
    w = np.zeros(N); idx = 0; cur = ratings[0]
    for t in range(1, N+1):
        if idx < len(RATING_TRIALS) and t == RATING_TRIALS[idx]:
            cur = ratings[idx]; idx += 1
        w[t-1] = cur
    return w

def softmax(beta, q):
    z = beta*(q - q.max()); e = np.exp(z); return e/e.sum()

def simulate(rng, alpha0, beta, phi, scenario, sched):
    """Return choices, rewards, optimal flags, and the craving regressor the model sees.
    Learning is always modulated by the TRUE craving; the model receives the OBSERVED one."""
    Q = np.zeros(3); ch = np.zeros(N, int); rw = np.zeros(N); opt = np.zeros(N, bool)

    if scenario in ("signal_clean", "noisy_wanting", "noise_null"):
        z = ar_ratings(rng)
        w_true = forward_fill(z)
        if scenario == "noisy_wanting":                       # noisy readout of the same craving
            zob = z + rng.normal(scale=1.17, size=len(z))     # gives corr(obs,true) ~ 0.65
            zob = (zob - zob.mean())/(zob.std()+1e-8)
            w_obs = forward_fill(zob)
        else:
            w_obs = w_true.copy()
        for t in range(N):
            c = rng.choice(3, p=softmax(beta, Q))
            r = WIN if rng.random() < sched[t, c] else LOSS
            a_t = sigmoid(logit(alpha0) + phi*w_true[t])
            Q[c] += a_t*(r - Q[c]); ch[t]=c; rw[t]=r; opt[t] = c == sched[t].argmax()
        return ch, rw, opt, w_obs

    # outcome_driven: craving is a leaky integrator of recent rewards, sampled at rating trials
    kappa, cscale = 0.7, 0.42                                  # cscale ~ SD of integrator, so modulation ~ unit scale
    c_int = 0.0; ratings = []; rset = set(RATING_TRIALS.tolist()); w_true = np.zeros(N)
    for t in range(N):
        if (t+1) in rset: ratings.append(c_int)               # craving reported before the trial
        w_true[t] = c_int/cscale
        c = rng.choice(3, p=softmax(beta, Q))
        r = WIN if rng.random() < sched[t, c] else LOSS
        a_t = sigmoid(logit(alpha0) + phi*w_true[t])
        Q[c] += a_t*(r - Q[c]); ch[t]=c; rw[t]=r; opt[t] = c == sched[t].argmax()
        c_int = kappa*c_int + (1-kappa)*r
    ratings = np.asarray(ratings); ratings = (ratings - ratings.mean())/(ratings.std()+1e-8)
    return ch, rw, opt, forward_fill(ratings)

def fit_core(ch, rw):
    """Grid maximum-likelihood over (alpha, beta), vectorized across all grid cells."""
    Q = np.zeros((CORE_A.size, 3)); ll = np.zeros(CORE_A.size)
    for t in range(N):
        lo = CORE_B[:,None]*Q; lo -= lo.max(1, keepdims=True)
        e = np.exp(lo); p = e/e.sum(1, keepdims=True)
        c = ch[t]; ll += np.log(p[:,c]+1e-12); Q[:,c] += CORE_A*(rw[t]-Q[:,c])
    i = ll.argmax(); return CORE_A[i], CORE_B[i], ll[i]

def fit_phi(ch, rw, w):
    """Grid maximum-likelihood over (alpha0, beta, phi) with alpha_t modulated by craving."""
    Q = np.zeros((PHI_A.size, 3)); ll = np.zeros(PHI_A.size)
    for t in range(N):
        lo = PHI_B[:,None]*Q; lo -= lo.max(1, keepdims=True)
        e = np.exp(lo); p = e/e.sum(1, keepdims=True)
        c = ch[t]; ll += np.log(p[:,c]+1e-12)
        a_t = sigmoid(PHI_LA + PHI_P*w[t]); Q[:,c] += a_t*(rw[t]-Q[:,c])
    i = ll.argmax(); return PHI_A[i], PHI_B[i], PHI_P[i], ll[i]

def run(n_agents, seeds):
    scen = ["signal_clean", "noisy_wanting", "outcome_driven", "noise_null"]
    rows = []
    for seed in seeds:
        for sc in scen:
            for a in range(n_agents):
                rng = np.random.default_rng((seed*9973 + hash(sc) % 9973)*100000 + a)
                ta = rng.uniform(0.08, 0.70)
                tb = float(np.exp(rng.uniform(np.log(1.0), np.log(8.0))))   # realistic, non-random betas
                tp = 0.0 if sc == "noise_null" else rng.uniform(-1.5, 1.5)
                sched = make_schedule(rng)
                ch, rw, opt, w = simulate(rng, ta, tb, tp, sc, sched)
                ca, cb, cll = fit_core(ch, rw)
                fa, fb, fp, fll = fit_phi(ch, rw, w)
                rows.append(dict(seed=seed, scenario=sc, agent=a, true_alpha=ta, true_beta=tb,
                                 true_phi=tp, core_a=ca, core_b=cb, phi_a=fa, phi_b=fb, rec_phi=fp,
                                 dll=fll-cll, acc=float(opt.mean()), final40=float(opt[-40:].mean())))
    return pd.DataFrame(rows)

def r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.nan if (np.std(x)==0 or np.std(y)==0) else float(np.corrcoef(x, y)[0,1])

def summarize(d):
    out = []
    for sc, g in d.groupby("scenario", sort=False):
        sign = np.mean(np.sign(g.true_phi)==np.sign(g.rec_phi)) if g.true_phi.std()>0 else np.nan
        det = lambda gg: float(np.mean(gg.dll > 2))
        row = dict(
            scenario=sc, n=len(g),
            core_alpha_r=r(g.true_alpha, g.core_a),
            core_beta_r=r(np.log(g.true_beta), np.log(g.core_b)),
            phi_alpha_r=r(g.true_alpha, g.phi_a),
            phi_beta_r=r(np.log(g.true_beta), np.log(g.phi_b)),
            phi_pearson=r(g.true_phi, g.rec_phi) if g.true_phi.std()>0 else np.nan,
            phi_spearman=(pd.Series(g.true_phi).rank().corr(pd.Series(g.rec_phi).rank())
                          if g.true_phi.std()>0 else np.nan),
            phi_sign_acc=sign,
            detect_dll_gt2=det(g),
            detect_smallphi=det(g[np.abs(g.true_phi)<0.5]) if sc!="noise_null" else np.nan,
            detect_bigphi=det(g[np.abs(g.true_phi)>=1.0]) if sc!="noise_null" else np.nan,
            mean_acc=g.acc.mean(), final40=g.final40.mean(),
        )
        out.append(row)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260615])
    ap.add_argument("--out", type=str, default=".")
    a = ap.parse_args()
    t0 = time.time()
    raw = run(a.n_agents, a.seeds)
    summ = summarize(raw)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    raw.to_csv(Path(a.out)/"recovery_v3_raw.csv", index=False)
    summ.to_csv(Path(a.out)/"recovery_v3_summary.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(summ.round(3).to_string(index=False))
    print("\nelapsed min %.2f | agents/scenario %d | seeds %s" %
          ((time.time()-t0)/60, a.n_agents, a.seeds))

if __name__ == "__main__":
    main()
