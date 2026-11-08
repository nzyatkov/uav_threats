"""Оценка слияния видимости: строки признаков по кэшу сценариев и метрики по ln V."""
import numpy as np
import pandas as pd
from .streams import DT
from .detectors_vis import VisFeatures, VIS_FEATS, physical_range_estimate
from .detectors_bvs5 import FEATS


def build_rows_vis(cache):
    X, lnV, meta = [], [], []
    for ci, c in enumerate(cache):
        fe = VisFeatures(c["epoch"])
        for m in c["msgs"]:
            x = fe.feed(m)
            if x is not None:
                X.append(x); lnV.append(float(np.log(c["S"]["V"][int(m["t_gen"] / DT)]))); meta.append((ci, m["t_gen"]))
    return np.array(X), np.array(lnV), meta


def ln_metrics(ln_hat, ln_true):
    e = np.abs(ln_hat - ln_true)
    return dict(med_abs_ln=float(np.median(e)), within_x1_5=float((e < np.log(1.5)).mean()), p90_abs_ln=float(np.percentile(e, 90)))


VBINS = [0, 300, 1000, 3000, 10000, 1e9]
VLAB = ["<0,3 км", "0,3-1 км", "1-3 км", "3-10 км", ">10 км"]


def by_bin(ln_hat, ln_true):
    V = np.exp(ln_true); idx = np.digitize(V, VBINS[1:-1]); out = {}
    for i, lab in enumerate(VLAB):
        m = idx == i
        if m.sum() > 20: out[lab] = float(np.median(np.abs(ln_hat[m] - ln_true[m])))
    return out


def physics_fusion(X, k_l=0.15, k_c=0.4, s_nsu=0.7, s_rng=0.3):
    """Байесовское слияние в лог-шкале: НСУ-3 (σ 0,7) + оценка по дальностям сенсоров (σ 0,3, если надёжна). Возвращает ln V̂."""
    j_n = VIS_FEATS.index("logV"); j_l = VIS_FEATS.index("d_first_lidar"); j_c = VIS_FEATS.index("d_first_cam"); j_a = VIS_FEATS.index("age_ratio")
    out = np.empty(len(X))
    for i, x in enumerate(X):
        v, ok = physical_range_estimate(x[j_l], x[j_c], x[j_a], k_l, k_c)
        w_n = 1 / s_nsu ** 2; num = w_n * x[j_n]; den = w_n
        if ok:
            w_r = 1 / s_rng ** 2; num += w_r * np.log(v); den += w_r
        out[i] = num / den
    return out


def events_vis(lnV_hat, meta, cache, thr, tau=10.0, n_on=3, n_off=5, post=60.0):
    """Событие «видимость < thr»: Gate по сглаженной (EMA, лог-шкала) оценке V̂. Метрики: recall, медианная задержка, ложных в час."""
    from .alerts import Gate
    from .eval_bvs5 import segments
    by = {}
    for (ci, t), v in zip(meta, lnV_hat): by.setdefault(ci, ([], []))[0].append(t); by[ci][1].append(v)
    n = hit = fa = 0; delays = []; hours = 0.0
    for ci, c in enumerate(cache):
        hours += c["duration"] / 3600.0
        ts, vs = by.get(ci, ([], [])); g = Gate(-1.0, -1.0, n_on, n_off); al = []; sm = None; tl = None
        lo, hi = np.log(thr), np.log(1.5 * thr)
        state = False; c_on = c_off = 0
        for t, v in zip(ts, vs):
            sm = v if sm is None else sm + (1 - np.exp(-(t - tl) / tau)) * (v - sm); tl = t
            if not state:
                c_on = c_on + 1 if sm < lo else 0
                if c_on >= n_on: state = True; c_on = 0; al.append(t)
            else:
                c_off = c_off + 1 if sm > hi else 0
                if c_off >= n_off: state = False; c_off = 0
        used = np.zeros(len(al), bool)
        for (t0, t1) in segments(c["S"]["V"] < thr):
            n += 1; a_in = [j for j, t in enumerate(al) if t0 <= t <= t1 + post]
            if a_in: hit += 1; delays.append(al[a_in[0]] - t0)
            for j in a_in: used[j] = True
        fa += int((~used).sum())
    return dict(событий=n, recall=hit / max(n, 1), задержка_медиана_с=float(np.median(delays)) if delays else np.nan, ложных_в_час=fa / hours)
