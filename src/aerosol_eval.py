"""Выборки сценариев и извлечение признаков (угрозы 1, 14, 15)."""
import numpy as np
import pandas as pd
from .synth_aerosol import AeroConfig, render, depth_profile, lidar_returns
from .aerosol_est import (tau_dcp, beta_from_rows, beta_lidar, spectral_features, vis_features,
                          anomaly_analysis, classify_rule, CLASSES)


def sample_cfg(rng, seed, label, terrain="veg", tau_plume=None):
    c = AeroConfig(seed=seed, terrain=terrain, alt=rng.uniform(150, 500),
                   phi_top=rng.uniform(3, 8), phi_bot=rng.uniform(28, 40), t_air=rng.uniform(5, 32))
    bg_clear = lambda: (rng.uniform(15000, 40000), "haze")
    bg_hazy = lambda: (rng.uniform(5000, 30000), rng.choice(["haze", "smoke"]))
    if label == "clear":
        c.vis_m, c.kind = bg_clear()
    elif label == "haze_fine":
        c.vis_m, c.kind = 10 ** rng.uniform(np.log10(800), np.log10(8000)), rng.choice(["haze", "smoke"])
    elif label == "dust_coarse":
        c.vis_m, c.kind = 10 ** rng.uniform(np.log10(800), np.log10(8000)), "dust"
    elif label in ("plume_fire", "plume_hidden"):
        c.vis_m, c.kind = bg_clear() if rng.random() < .5 else bg_hazy()
        c.plume, c.hotspot = True, label == "plume_fire"
        c.plume_tau = tau_plume if tau_plume is not None else rng.uniform(0.5, 3.0)
        c.plume_len = rng.uniform(35, 60)
    elif label == "dust_front":
        c.vis_m, c.kind = rng.uniform(10000, 40000), "dust"
        c.front, c.front_tau = True, rng.uniform(0.6, 2.0)
    elif label == "hot_decoy":
        c.vis_m, c.kind = bg_clear() if rng.random() < .5 else bg_hazy()
        c.hot_decoy = True
    return c


def extract(cfg, rng, tele_err=0.08, with_lidar=True):
    """Все признаки одного кадра; телеметрия (высота, углы) известна с относительной ошибкой tele_err."""
    r = render(cfg)
    alt_hat = cfg.alt * (1 + rng.normal(0, tele_err))
    d = depth_profile(alt_hat, cfg.phi_top, cfg.phi_bot)
    t_air_hat = cfg.t_air + rng.normal(0, 0.5)
    tau, A = tau_dcp(r["rgb"])
    f = dict(label=None)
    f.update(vis_features(r["rgb"], r["T"], d))
    f.update(spectral_features(r["rgb"], r["T"], d, t_air_hat))
    an, resid = anomaly_analysis(r["rgb"], r["T"], d)
    f.update({f"an_{k}": v for k, v in an.items()})
    if with_lidar:
        R, I, _ = lidar_returns(cfg.vis_m, cfg.kind, seed=cfg.seed + 33)
        f["beta_lidar"] = beta_lidar(R, I)
    f["true_beta"] = 3.912 / cfg.vis_m
    f["true_V"] = cfg.vis_m
    f["terrain"] = cfg.terrain
    f["kind"] = cfg.kind
    return f, r


def build_table(labels, n_per, seed0, terrain="veg", **kw):
    rng = np.random.default_rng(seed0)
    rows = []
    for lab in labels:
        for i in range(n_per):
            cfg = sample_cfg(rng, seed0 * 1000 + len(rows), lab, terrain, **kw)
            f, _ = extract(cfg, rng)
            f["label"] = lab
            rows.append(f)
    return pd.DataFrame(rows)


def prepare(labels, n_per, seed0, terrain="veg", **kw):
    """Отрендерить сцены один раз для перебора порогов: [(label, render, depth_hat, cfg)]."""
    rng = np.random.default_rng(seed0)
    items = []
    for lab in labels:
        for i in range(n_per):
            cfg = sample_cfg(rng, seed0 * 1000 + len(items), lab, terrain, **kw)
            r = render(cfg)
            d = depth_profile(cfg.alt * (1 + rng.normal(0, 0.08)), cfg.phi_top, cfg.phi_bot)
            items.append((lab, r, d, cfg))
    return items


def ref_rgb(cfg):
    """Эталонный кадр той же местности из прошлого пролёта: чистая атмосфера, другой шум (карта опорного состояния)."""
    c = AeroConfig(seed=cfg.seed, terrain=cfg.terrain, alt=cfg.alt, phi_top=cfg.phi_top, phi_bot=cfg.phi_bot,
                   t_air=cfg.t_air, vis_m=30000.0, kind="haze", noise_offset=7)
    return render(c)["rgb"]


def classify_items(items, thr=0.25, maf=0.03, elong=2.0, rule_kw=None, use_ref=False, ref_diff=False):
    out = []
    for lab, r, d, cfg in items:
        an, _ = anomaly_analysis(r["rgb"], r["T"], d, thr=thr, min_area_frac=maf, I_ref=ref_rgb(cfg) if use_ref else None, ref_diff=ref_diff)
        sp = spectral_features(r["rgb"], r["T"], d, cfg.t_air)
        out.append(classify_rule(an, sp, plume_elong=elong, comp_min=maf, **(rule_kw or {})))
    return np.array(out)


def vis_table(n, seed0, terrain, v_lo=500, v_hi=40000):
    """Сцены без шлейфов: V log-равномерно, тип аэрозоля случайный."""
    rng = np.random.default_rng(seed0)
    rows = []
    for i in range(n):
        cfg = AeroConfig(seed=seed0 * 1000 + i, terrain=terrain, alt=rng.uniform(150, 500), phi_top=rng.uniform(3, 8),
                         phi_bot=rng.uniform(28, 40), t_air=rng.uniform(5, 32),
                         vis_m=float(np.exp(rng.uniform(np.log(v_lo), np.log(v_hi)))), kind=str(rng.choice(["haze", "smoke", "dust"])))
        f, _ = extract(cfg, rng)
        f["coarse"] = cfg.kind == "dust"
        f["V_ext"] = cfg.vis_m * float(np.exp(rng.normal(0, 0.6)))    # внешний прогноз НСУ-3 (допущение: σ_ln = 0.6)
        rows.append(f)
    return pd.DataFrame(rows)
