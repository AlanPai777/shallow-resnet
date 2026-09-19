"""Small numpy-only statistics helpers (no scipy needed).

Used by stats.py and make_figures.py. Everything here is written for the balanced 2x2 design of the
report: 2 depths x residual on/off, 3 seeds per cell.
"""
import itertools
import math

import numpy as np


# ---- regularized incomplete beta function (continued fraction, Numerical Recipes) ----
def _betacf(a, b, x, maxit=500, eps=3e-14):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return betai(df / 2.0, 0.5, df / (df + t * t))


def f_sf(F, d1, d2):
    """Upper-tail probability of the F distribution."""
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))


def t_crit(df, alpha=0.05):
    """Two-sided critical value of Student's t, by bisection."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---- two-sample comparisons ----
def welch(a, b):
    """Welch t-test of mean(a) - mean(b), with a t-based 95% confidence interval."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    t = (a.mean() - b.mean()) / se
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tc = t_crit(df)
    diff = a.mean() - b.mean()
    return dict(diff=diff, se=se, t=t, df=df, p=t_two_sided_p(t, df), lo=diff - tc * se, hi=diff + tc * se)


def permutation_p(a, b):
    """Exact two-sided permutation p-value for a difference in means (smallest possible: 0.10 for 3 vs 3)."""
    allv = np.concatenate([a, b])
    n = len(a)
    obs = abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(len(allv)), n):
        mask = np.zeros(len(allv), bool)
        mask[list(idx)] = True
        total += 1
        count += abs(allv[mask].mean() - allv[~mask].mean()) >= obs - 1e-12
    return count / total


def cohens_d(a, b):
    return (a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


# ---- 2x2 interaction ----
def anova_2x2(acc):
    """Two-way ANOVA on {(depth8|depth20, residual on|off): array of 3 values}.

    Pooled error term (8 residual df). Returns F/p for residual, depth and their interaction, and the
    difference-in-differences (depth-20 effect minus depth-8 effect) with a 95% CI.
    """
    n = 3
    cm = {k: v.mean() for k, v in acc.items()}
    allv = np.concatenate(list(acc.values()))
    g = allv.mean()
    on = np.concatenate([acc[(8, True)], acc[(20, True)]]).mean()
    off = np.concatenate([acc[(8, False)], acc[(20, False)]]).mean()
    d8 = np.concatenate([acc[(8, True)], acc[(8, False)]]).mean()
    d20 = np.concatenate([acc[(20, True)], acc[(20, False)]]).mean()
    ss_res = 6 * ((on - g) ** 2 + (off - g) ** 2)
    ss_dep = 6 * ((d8 - g) ** 2 + (d20 - g) ** 2)
    ss_cells = n * sum((cm[k] - g) ** 2 for k in cm)
    ss_int = ss_cells - ss_res - ss_dep
    ss_err = sum(((acc[k] - cm[k]) ** 2).sum() for k in acc)
    mse = ss_err / 8
    did = (cm[(20, True)] - cm[(20, False)]) - (cm[(8, True)] - cm[(8, False)])
    se_did = math.sqrt(mse * 4 / 3)
    tc = t_crit(8)
    return dict(
        mse=mse, did=did, did_lo=did - tc * se_did, did_hi=did + tc * se_did,
        F_res=ss_res / mse, p_res=f_sf(ss_res / mse, 1, 8),
        F_dep=ss_dep / mse, p_dep=f_sf(ss_dep / mse, 1, 8),
        F_int=ss_int / mse, p_int=f_sf(ss_int / mse, 1, 8),
        cell_vars=[float(acc[k].var(ddof=1)) for k in acc],
    )


def unpooled_did(acc):
    """Difference-in-differences with Welch-Satterthwaite degrees of freedom (no pooled variance)."""
    cells = [acc[(20, True)], acc[(20, False)], acc[(8, True)], acc[(8, False)]]
    signs = [1, -1, -1, 1]
    did = sum(s * c.mean() for s, c in zip(signs, cells))
    vi = [c.var(ddof=1) / len(c) for c in cells]
    se = math.sqrt(sum(vi))
    df = sum(vi) ** 2 / sum(v ** 2 / (len(c) - 1) for v, c in zip(vi, cells))
    t = did / se
    tc = t_crit(df)
    return dict(did=did, se=se, t=t, df=df, p=t_two_sided_p(t, df), lo=did - tc * se, hi=did + tc * se)
