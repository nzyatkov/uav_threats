"""Генерация выборок последовательностей для оценки (угрозы 10, 16)."""
import numpy as np
from .synth_light import SeqConfig, make_sequence
from .light_features import seq_features

BENIGN_TYPES = ["night_clean", "night_city", "day_clean", "day_glint", "night_fire", "day_fire"]


def benign_cfg(rng, seed, kind=None):
    kind = kind or BENIGN_TYPES[rng.integers(len(BENIGN_TYPES))]
    c = SeqConfig(seed=seed)
    if kind == "night_clean": pass
    elif kind == "night_city": c.urban = rng.uniform(0.1, 1.0)
    elif kind == "day_clean": c.night = False
    elif kind == "day_glint": c.night = False; c.glint = True; c.glint_I = 10 ** rng.uniform(0.5, 1.3); c.glint_start = int(rng.integers(25, 50))
    elif kind == "night_fire": c.fire = True; c.fire_I = 10 ** rng.uniform(0, 2); c.urban = rng.uniform(0, 0.6); c.fire_start = int(rng.integers(25, 50))
    elif kind == "day_fire": c.night = False; c.fire = True; c.fire_I = 10 ** rng.uniform(0.5, 2); c.fire_start = int(rng.integers(25, 50))
    return kind, c


def laser_cfg(rng, seed, I=None, night=None, core=None):
    night = bool(rng.random() < 0.7) if night is None else night
    return SeqConfig(seed=seed, night=night, laser=True,
                     laser_I=I if I is not None else 10 ** rng.uniform(1, 3.5),
                     laser_core=core if core is not None else rng.uniform(0.8, 3.0),
                     laser_spots=int(rng.integers(1, 4)), laser_duty=1.0 if rng.random() < 0.2 else rng.uniform(0.4, 0.8),   # 20 % непрерывных лазеров
                     laser_onset=int(rng.integers(25, 45)), laser_len=int(rng.integers(20, 40)),
                     urban=rng.uniform(0, 0.6) if night else 0.0)


def build(cfgs):
    """[(kind, cfg)] -> список (kind, cfg, features DataFrame, labels)."""
    out = []
    for kind, cfg in cfgs:
        s = make_sequence(cfg)
        out.append((kind, cfg, seq_features(s), s["labels"]))
    return out


# ---------------- оценка детекторов угрозы 10 ----------------
import pandas as pd
from sklearn.metrics import roc_auc_score
from .light_detectors import rule_laser, seq_event_outcome

WARM = 15
DETECTORS = ["R0_sat_jump", "R1_thermal_gate", "R2_pulse_confirm", "R3_day_context", "IForest", "PCA_AE"]


def run_detectors(F, cfg, am, k_sd=8.0, day_factor=4.0):
    """Все детекторы на одной последовательности: {имя: (alarm, score)}."""
    day = not cfg.night
    out = {}
    out["R0_sat_jump"] = rule_laser(F, use_thermal=False, k_sd=k_sd, day=False)
    out["R1_thermal_gate"] = rule_laser(F, use_thermal=True, k_sd=k_sd, day=False)
    out["R2_pulse_confirm"] = rule_laser(F, use_thermal=True, k_sd=k_sd, pulse_confirm=True, day=False)
    out["R3_day_context"] = rule_laser(F, use_thermal=True, k_sd=k_sd, pulse_confirm=True, day=day, day_factor=day_factor)
    for k, name in (("iforest", "IForest"), ("pca", "PCA_AE")):
        a, s = am.alarm(F, k)
        a[:WARM] = False
        out[name] = (a, s)
    return out


def evaluate_detectors(items, am, **kw):
    """items - результат build(). Возвращает DataFrame по последовательностям (flag, delay, kind, параметры)."""
    rows = []
    for kind, cfg, F, lab in items:
        res = run_detectors(F, cfg, am, **kw)
        for name, (a, s) in res.items():
            if cfg.laser:
                ok, d = seq_event_outcome(a, lab["laser"], cfg.laser_onset)
            else:
                ok, d = bool(a[WARM:].any()), np.nan
            rows.append(dict(det=name, kind=kind, flag=ok, delay=d, laser=cfg.laser, night=cfg.night,
                             I=cfg.laser_I, core=cfg.laser_core, cw=cfg.laser_duty >= 1.0, urban=cfg.urban))
    return pd.DataFrame(rows)


def frame_auroc(items, am, **kw):
    """AUROC по кадрам: кадры события лазера против кадров безобидных последовательностей (после прогрева)."""
    ys, sc = [], {n: [] for n in DETECTORS}
    for kind, cfg, F, lab in items:
        res = run_detectors(F, cfg, am, **kw)
        y = lab["laser"][WARM:].astype(int) if cfg.laser else np.zeros(len(F) - WARM, int)
        ys.append(y)
        for n, (a, s) in res.items():
            sc[n].append(np.asarray(s)[WARM:])
    y = np.concatenate(ys)
    return {n: roc_auc_score(y, np.concatenate(v)) for n, v in sc.items()}
