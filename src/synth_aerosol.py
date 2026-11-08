"""Синтетика аэрозольной деградации видимости (угрозы 1, 14, 15) с наклонной геометрией съёмки.

Модель переноса излучения (Koschmieder / He et al. 2009), по каналам c ∈ {R,G,B}:
    I_c = J_c * exp(-τ_c) + A_c * (1 - exp(-τ_c)),   τ_c = β_c * d(y) + τ_extra,c(y,x)
β_550 = 3.912 / V (V - метеорологическая дальность видимости), β_c = β_550 * (550/λ_c)^α (Ангстрем).
Тип аэрозоля: α (спектральный наклон), κ = β_LWIR/β_550 (ослабление в ЛВИК), цвет засветки A.
  мелкий (дым, смог, дымка): α≈1.3-1.6, κ≈0.03;  крупный (пыль, песок): α≈0.3, κ≈0.25 (ДОПУЩЕНИЕ по порядку величины).
Местные неоднородности: шлейф дыма от очага (экспоненциальный спад вдоль ветра, расширение) и пылевой фронт (стена с прямой кромкой).
Тепловизор: T_obs = T_g*exp(-κτ) + T_air*(1-exp(-κτ)); очаг +300 К; «горячие камни» - ложные тепловые пятна.
LiDAR (ближний ИК, 905 нм): интенсивность ∝ ρ*exp(-2β_L R)/R^2.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from .synth_fog import make_texture

H, W = 96, 128
WL = np.array([650.0, 550.0, 470.0])          # R, G, B, нм
AEROSOL = {
    "smoke": dict(alpha=1.6, kappa=0.03, A=(0.60, 0.56, 0.52)),
    "haze":  dict(alpha=1.3, kappa=0.03, A=(0.72, 0.73, 0.76)),
    "dust":  dict(alpha=0.3, kappa=0.25, A=(0.80, 0.70, 0.52)),
}
PALETTE = {"forest": (0.10, 0.25, 0.08), "grass": (0.25, 0.42, 0.12), "soil": (0.40, 0.30, 0.20),
           "rock": (0.50, 0.50, 0.50), "water": (0.08, 0.18, 0.35), "sand": (0.82, 0.70, 0.50)}
TERRAIN_MIX = {"veg": [("forest", .45), ("grass", .25), ("soil", .15), ("water", .10), ("rock", .05)],
               "sand": [("sand", .55), ("rock", .25), ("soil", .15), ("grass", .05)],
               "mixed": [("sand", .25), ("grass", .25), ("forest", .2), ("rock", .15), ("soil", .15)]}


@dataclass
class AeroConfig:
    seed: int = 0
    terrain: str = "veg"
    alt: float = 300.0            # высота над землёй, м (БВС-1)
    phi_top: float = 5.0          # угол визирования верхней строки кадра ниже горизонта, град
    phi_bot: float = 35.0
    vis_m: float = 40000.0        # метеорологическая дальность видимости фонового аэрозоля
    kind: str = "haze"            # smoke | haze | dust
    plume: bool = False
    plume_tau: float = 1.5        # пиковая оптическая толщина шлейфа у источника
    plume_len: float = 50.0       # px
    hotspot: bool = True          # виден ли очаг (тепловая аномалия) у источника шлейфа
    hot_decoy: bool = False       # ложные горячие пятна (нагретые камни)
    front: bool = False
    front_tau: float = 1.2
    t_air: float = 22.0
    noise: float = 0.01
    noise_offset: int = 0         # смещение зерна шума (эталонный кадр той же местности с другим шумом)


def depth_profile(alt, phi_top, phi_bot, h=H):
    phi = np.radians(np.linspace(phi_top, phi_bot, h))
    return alt / np.sin(phi)                    # наклонная дальность до поверхности для каждой строки, м


def make_scene(seed, terrain):
    rng = np.random.default_rng(seed)
    t1 = gaussian_filter(make_texture(size=W, seed=seed)[:H], 1.5)
    r = (t1.ravel().argsort().argsort() / t1.size).reshape(t1.shape)   # ранги 0..1
    names, probs = zip(*TERRAIN_MIX[terrain]); edges = np.cumsum(probs); edges[-1] = 1.01
    cls = np.searchsorted(edges, r)
    J = np.zeros((H, W, 3))
    for i, n in enumerate(names):
        J[cls == i] = PALETTE[n]
    detail = make_texture(size=W, seed=seed + 5)[:H]
    J = np.clip(J * (0.75 + 0.5 * detail[..., None]) + rng.normal(0, 0.01, J.shape), 0, 1)
    J = gaussian_filter(J, (0.7, 0.7, 0))
    Tg_tex = make_texture(size=W, seed=seed + 9)[:H]
    return J, 3.0 + 6.0 * (Tg_tex - 0.5), cls, names


def _plume_tau(rng, tau0, L, shape=(H, W)):
    sy, sx = rng.uniform(0.5 * H, 0.85 * H), rng.uniform(0.2 * W, 0.8 * W)
    th = np.radians(rng.uniform(-60, 60))
    u = np.array([-np.cos(th), np.sin(th)])     # (dy, dx): вверх по кадру со сносом
    n = np.array([u[1], -u[0]])
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dy, dx = yy - sy, xx - sx
    s = dy * u[0] + dx * u[1]; c = dy * n[0] + dx * n[1]
    sg = 3.0 + 0.12 * np.clip(s, 0, None)
    tau = tau0 * np.exp(-np.clip(s, 0, None) / L) * np.exp(-c ** 2 / (2 * sg ** 2)) * (s > -2)
    return tau, (sy, sx)


def _front_tau(rng, tau0, shape=(H, W)):
    a = np.radians(rng.uniform(0, 180))
    q = np.mgrid[0:shape[0], 0:shape[1]]
    proj = (q[0] - shape[0] / 2) * np.sin(a) + (q[1] - shape[1] / 2) * np.cos(a)
    off = rng.uniform(-15, 15)
    f = tau0 / (1 + np.exp(-(proj - off) / 4.0))
    return f * (1 + 0.15 * gaussian_filter(rng.normal(size=shape), 6) * 5)


def render(cfg: AeroConfig):
    rng = np.random.default_rng(cfg.seed + 101)
    rn = np.random.default_rng(cfg.seed + 303 + cfg.noise_offset)
    J, Tg_off, cls, names = make_scene(cfg.seed, cfg.terrain)
    d = depth_profile(cfg.alt, cfg.phi_top, cfg.phi_bot)[:, None]
    ae = AEROSOL[cfg.kind]
    b550 = 3.912 / cfg.vis_m
    tau550_bg = b550 * d * np.ones((H, W))
    tau_p = np.zeros((H, W)); tau_f = np.zeros((H, W)); src = None
    if cfg.plume:
        tau_p, src = _plume_tau(rng, cfg.plume_tau, cfg.plume_len)
    if cfg.front:
        tau_f = _front_tau(rng, cfg.front_tau)
    sm, du = AEROSOL["smoke"], AEROSOL["dust"]
    spec = lambda a: (550.0 / WL) ** a          # (3,)
    tau_c = (tau550_bg[..., None] * spec(ae["alpha"]) + tau_p[..., None] * spec(sm["alpha"])
             + tau_f[..., None] * spec(du["alpha"]))
    tot550 = tau550_bg + tau_p + tau_f + 1e-9
    A = (tau550_bg[..., None] * np.array(ae["A"]) + tau_p[..., None] * np.array(sm["A"])
         + tau_f[..., None] * np.array(du["A"])) / tot550[..., None]
    tr = np.exp(-tau_c)
    I = J * tr + A * (1 - tr)
    I = np.clip(I + rn.normal(0, cfg.noise, I.shape), 0, 1)
    # тепловизор
    kap_tau = ae["kappa"] * tau550_bg + sm["kappa"] * tau_p + du["kappa"] * tau_f
    Tg = cfg.t_air + Tg_off
    hot = np.zeros((H, W)); hot_pos = []
    yy, xx = np.mgrid[0:H, 0:W]
    if cfg.plume and cfg.hotspot:
        hot += 300.0 * np.exp(-((yy - src[0]) ** 2 + (xx - src[1]) ** 2) / (2 * 1.5 ** 2)); hot_pos.append(src)
    if cfg.hot_decoy:
        for _ in range(int(rng.integers(1, 4))):
            py, px = rng.uniform(8, H - 8), rng.uniform(8, W - 8)
            hot += rng.uniform(25, 60) * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * rng.uniform(1.5, 3.0) ** 2))
    Tt = (Tg + hot) * np.exp(-kap_tau) + cfg.t_air * (1 - np.exp(-kap_tau)) + rn.normal(0, 0.15, (H, W))
    return dict(rgb=I, T=Tt, tau550=tau550_bg + tau_p + tau_f, tau_p=tau_p, tau_f=tau_f, src=src,
                depth=d[:, 0], cfg=cfg, J=J)


def lidar_returns(vis_m, kind="haze", seed=0, n=1500, r_max=400.0, i_min=1e-7):
    """Облако возвратов 905 нм: R, I. Возвраты ниже порога обнаружения отбрасываются (цензурирование)."""
    rng = np.random.default_rng(seed)
    ae = AEROSOL[kind]
    bL = 3.912 / vis_m * (550.0 / 905.0) ** ae["alpha"]
    R = rng.uniform(30, r_max, n)
    rho = np.exp(rng.normal(np.log(0.3), 0.5, n))
    I = rho * np.exp(-2 * bL * R) / R ** 2 * np.exp(rng.normal(0, 0.15, n))
    keep = I > i_min
    return R[keep], I[keep], bL
