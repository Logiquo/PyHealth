# Contributor: Neil Hajela (nhajela2@illinois.edu)
"""Historical SUPPORT metrics for benchmark comparison.

Preserves the original evaluator's test-cohort censoring KM interpolation,
strict concordance comparisons and 1000-point uniform grid. These helpers
are not proposed as a general-purpose PyHealth survival metric API.
"""

from __future__ import annotations

import numpy as np
import torch


def km_fit(times, event_censor):
    """Fit a Kaplan-Meier curve; returns step times and survival values."""
    times = np.asarray(times, float)
    ev = np.asarray(event_censor, np.int8)
    uniq = np.unique(times)
    surv = []
    s = 1.0
    for u in uniq:
        at_risk = np.sum(times >= u)
        d = np.sum((times == u) & (ev == 1))
        s *= 1 - d / at_risk
        surv.append(s)
    return np.r_[0.0, uniq], np.r_[1.0, np.asarray(surv)]


def km_predict(kmt, kms, q):
    """Linearly interpolate a fitted Kaplan-Meier curve at times ``q``."""
    return np.interp(np.asarray(q, float), kmt, kms, left=1.0, right=kms[-1])


@torch.no_grad()
def survival_matrix(model, x, times, batch=512, time_chunk=250):
    """Return model survival probabilities with shape ``[time, patient]``."""
    model.eval()
    times = np.asarray(times, np.float32)
    all_batches = []
    for s in range(0, len(x), batch):
        params = model.network.mixture_parameters(
            torch.as_tensor(x[s : s + batch], dtype=torch.float32)
        )
        chunks = []
        for j in range(0, len(times), time_chunk):
            chunks.append(
                model.network.survival_at(
                    params,
                    torch.as_tensor(times[j : j + time_chunk], dtype=torch.float32),
                )
                .cpu()
                .numpy()
            )
        all_batches.append(np.concatenate(chunks, axis=1))
    return np.concatenate(all_batches, axis=0).T


def _sorted_split(split):
    """Return durations, events and features sorted by duration."""
    order = np.argsort(split.duration)
    t = np.asarray(split.duration, float)[order]
    e = np.asarray(split.event, float)[order]
    x = np.asarray(split.x, np.float32)[order]
    return t, e, x


def _concordance(t, e, H, pos, G_T, eps):
    """IPCW time-dependent concordance, skipping events with ``G(T) < eps``."""
    G = G_T.copy()
    G[G == 0] = eps / 2
    num = 0.0
    score = 0.0
    for i in np.flatnonzero(e > 0):
        if G[i] < eps:
            continue
        idx = (t > t[i]) | ((t == t[i]) & (e == 0))
        w = 1 / (G[i] ** 2)
        score += np.sum(H[pos[i], idx] < H[pos[i], i]) * w
        num += np.sum(idx) * w
    return float(score / num)


def _brier_and_log_loss(model, x, t, e, kmt, kms, G_T, horizon_eps):
    """IPCW integrated Brier score and binomial log-likelihood.

    The grid runs from the shortest duration to the last duration whose
    censoring survival exceeds ``horizon_eps``.
    """
    grid = np.linspace(t.min(), float(np.max(t[G_T > horizon_eps])), 1000)
    S = survival_matrix(model, x, grid)
    G_t = km_predict(kmt, kms, grid)
    ind = (t[None, :] <= grid[:, None]).astype(float)

    mr = G_t > 1e-8
    S = S[mr]
    ind = ind[mr]
    Gg = G_t[mr]
    mc = G_T > 1e-8
    S = S[:, mc]
    ind = ind[:, mc]
    lab = e[mc][None, :]
    GT = G_T[mc][None, :]
    Gg = Gg[:, None]

    ibs = float(np.mean(S**2 * lab * ind / GT + (1 - S) ** 2 * (1 - ind) / Gg))
    ibll = float(
        np.mean(
            np.log(1 - S + 1e-10) * lab * ind / GT + np.log(S + 1e-10) * (1 - ind) / Gg
        )
    )
    return ibs, ibll


def _prepare(model, split):
    """Compute the inputs shared by every metric for one test subset."""
    t, e, x = _sorted_split(split)
    kmt, kms = km_fit(t, 1 - e)
    G_T = km_predict(kmt, kms, t)
    eval_t = np.unique(t)
    S_event = survival_matrix(model, x, eval_t)
    H = -np.log(np.clip(S_event, 1e-30, 1.0))
    pos = np.searchsorted(eval_t, t)
    return t, e, x, kmt, kms, G_T, H, pos


def paper_metrics(model, split):
    """Concordance, IBS and IBLL at the paper's three censoring horizons."""
    t, e, x, kmt, kms, G_T, H, pos = _prepare(model, split)
    out = {}
    for eps, key in [(1e-8, "c"), (0.2, "c2"), (0.4, "c4")]:
        out[key] = _concordance(t, e, H, pos, G_T, eps)
    for eps, suffix in [(0, ""), (0.2, "2"), (0.4, "4")]:
        ibs, ibll = _brier_and_log_loss(model, x, t, e, kmt, kms, G_T, eps)
        out["ibs" + suffix] = ibs
        out["ibll" + suffix] = ibll
    return out


@torch.no_grad()
def primary_metrics(model, split):
    """Primary paper metrics at the near-untruncated censoring horizon."""
    t, e, x, kmt, kms, G_T, H, pos = _prepare(model, split)
    c = _concordance(t, e, H, pos, G_T, 1e-8)
    ibs, ibll = _brier_and_log_loss(model, x, t, e, kmt, kms, G_T, 0)
    return {"c": c, "ibs": ibs, "ibll": ibll}


def tail_diagnostics(model, split, train_split):
    """Mean predicted survival at and beyond the training time range."""
    x = np.asarray(split.x, np.float32)
    t_test = np.asarray(split.duration, float)
    t_train = np.asarray(train_split.duration, float)
    max_train = float(np.max(t_train))
    q99_train = float(np.quantile(t_train, 0.99))
    max_test = float(np.max(t_test))
    probe = np.array([q99_train, max_train, 2.0 * max_train], np.float32)
    S = survival_matrix(model, x, probe)
    return {
        "tail_q99train_meanS": float(S[0].mean()),
        "tail_maxtrain_meanS": float(S[1].mean()),
        "tail_2xmaxtrain_meanS": float(S[2].mean()),
        "max_train_time": max_train,
        "max_test_time": max_test,
    }
