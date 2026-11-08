"""Детекторы запотевания (угроза 28): физический (по БВС-5) и видео (по резкости), слияние, оценка."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .detectors import lag_filter, hysteresis, cusum_drop
from .metrics import first_true_time
from .synth_fog import simulate, random_scenario


def physics_detector(df, tau_hat=120.0, offset_hat=0.0, margin_K=0.5, sigma_K=0.5,
                     on=0.6, off=0.3, dt=1.0):
    """Предсказание запотевания по T/RH снаружи (линза не измеряется).

    T_lens оценивается звеном первого порядка по измеренной T воздуха.
    drive = Td_meas - T_lens_hat (>0 => конденсат). Предупреждаем заранее: при drive > -margin.
    p = Phi((drive + margin)/sigma): sigma учитывает шум датчиков и неточность модели.
    """
    lens_hat = lag_filter(df["T_meas"].values + offset_hat, tau_hat, dt)
    drive = df["Td_meas"].values - lens_hat
    p = norm.cdf((drive + margin_K) / sigma_K)
    return pd.DataFrame(dict(drive=drive, p_phys=p, lens_hat=lens_hat,
                             warn_phys=hysteresis(p, on, off)))


def video_detector(sharp, k=0.3, h=4.0, alpha=0.01, warmup=30):
    """CUSUM по падению log-резкости относительно медленной базы (база замораживается при тревоге)."""
    ls = np.log(np.asarray(sharp))
    base = np.empty_like(ls)
    base[0] = ls[0]
    alarm = np.zeros(len(ls), dtype=bool)
    S = np.zeros(len(ls))
    on = False
    for i in range(1, len(ls)):
        z = base[i - 1] - ls[i]
        S[i] = min(max(0.0, S[i - 1] + z - k), 2.0 * h)   # ограничение сверху: без windup тревога гаснет после события
        if i > warmup:
            if not on and S[i] > h:
                on = True
            elif on and S[i] < h * 0.25:
                on = False
        alarm[i] = on
        base[i] = base[i - 1] if on else base[i - 1] + alpha * (ls[i] - base[i - 1])
    return pd.DataFrame(dict(S_video=S, alarm_video=alarm, base=base))


def run_pipeline(df, **phys_kw):
    ph = physics_detector(df, **phys_kw)
    vd = video_detector(df["sharp"].values)
    out = pd.concat([df.reset_index(drop=True), ph, vd], axis=1)
    out["alarm_fused"] = out["warn_phys"] | out["alarm_video"]       # любой из каналов
    out["confirmed"] = out["warn_phys"] & out["alarm_video"]         # оба канала
    return out


def event_stats(res, channels=("warn_phys", "alarm_video", "alarm_fused"),
                lead_window=300.0, lag_window=120.0):
    """Метрики одного прогона. onset = первый момент label=True."""
    t = res["t"].values
    has_event = bool(res["label"].any())
    onset = float(t[res["label"].values][0]) if has_event else np.nan
    row = dict(has_event=has_event, onset=onset)
    for ch in channels:
        a = res[ch].values
        if has_event:
            ta = first_true_time(a, t, onset - lead_window, onset + lag_window)
            row[f"{ch}_delay"] = ta - onset if not np.isnan(ta) else np.nan   # <0 = раньше события
            row[f"{ch}_detected"] = not np.isnan(ta)
        else:
            row[f"{ch}_false_alarm"] = bool(a.any())
            row[f"{ch}_fa_frac"] = float(a.mean())
    return row


def monte_carlo(n=200, seed=0, **phys_kw):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sc = random_scenario(rng, seed=seed * 100000 + i)
        res = run_pipeline(simulate(sc), **phys_kw)
        r = event_stats(res)
        r.update(tau=sc.tau, offset=sc.lens_offset)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(mc, channels=("warn_phys", "alarm_video", "alarm_fused")):
    pos = mc[mc.has_event]
    neg = mc[~mc.has_event]
    out = []
    for ch in channels:
        out.append(dict(
            channel=ch,
            recall=pos[f"{ch}_detected"].mean(),
            median_delay_s=pos[f"{ch}_delay"].median(),
            false_alarm_rate=neg[f"{ch}_false_alarm"].mean() if len(neg) else np.nan,
            n_pos=len(pos), n_neg=len(neg)))
    return pd.DataFrame(out).set_index("channel")
