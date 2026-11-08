"""Синтетика влияния осадков/облачности на видео, LiDAR и IMU (угрозы 5, 7, 8, 9).

Видео: мир (цветной холст) смещается на VX px/кадр; атмосфера I = J T + A(1-T), T = exp(-β d(y)), β = 3.912/V;
       осадки - частицы в воздухе (случайные в каждом кадре: полосы дождя, хлопья снега, крупные быстрые градины);
       капли на линзе - статичные размытые пятна (в координатах кадра); запотевание линзы - размытие.
LiDAR: лучи на угловой сетке (дальность до земли падает с углом), точки-помехи осадков заменяют первый возврат
       на малой дальности (интенсивность дождя ниже, снега выше), препятствия - угловые окна с меньшей дальностью.
IMU: вертикальное ускорение 200 Гц: турбулентность (низкочастотная) + шум вибраций + удары града (затухающие синусоиды).
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt
from .synth_fog import make_texture
from .synth_aerosol import PALETTE, TERRAIN_MIX, depth_profile

H, W = 96, 128
VX = 1.0
CW = 128 + 40


def make_world(seed, terrain="veg"):
    t1 = gaussian_filter(make_texture(size=CW, seed=seed)[:H], 1.5)
    r = (t1.ravel().argsort().argsort() / t1.size).reshape(t1.shape)
    names, probs = zip(*TERRAIN_MIX[terrain]); edges = np.cumsum(probs); edges[-1] = 1.01
    cls = np.searchsorted(edges, r)
    J = np.zeros((H, CW, 3))
    for i, n in enumerate(names):
        J[cls == i] = PALETTE[n]
    detail = make_texture(size=CW, seed=seed + 5)[:H]
    rng = np.random.default_rng(seed)
    J = np.clip(J * (0.7 + 0.6 * detail[..., None]) + rng.normal(0, 0.012, J.shape), 0, 1)
    return gaussian_filter(J, (0.6, 0.6, 0))


@dataclass
class WxConfig:
    n: int = 24
    seed: int = 0
    terrain: str = "veg"
    alt: float = 300.0
    phi_top: float = 5.0
    phi_bot: float = 35.0
    vis_m: float = 30000.0        # дальность видимости (облако/туман: 30-300 м)
    rain: float = 0.0             # интенсивность 0..1
    snow: float = 0.0
    hail: float = 0.0
    drops: int = 0                # число капель на линзе
    lens_fog: float = 0.0         # степень запотевания линзы 0..1
    noise: float = 0.012


def _splat(img, y, x, sig, amp):
    r = int(3 * sig) + 1
    y0, x0 = int(round(y)), int(round(x))
    if y0 < -r or y0 > H + r or x0 < -r or x0 > W + r:
        return
    ys = slice(max(0, y0 - r), min(H, y0 + r + 1)); xs = slice(max(0, x0 - r), min(W, x0 + r + 1))
    yy, xx = np.mgrid[ys, xs]
    g = amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sig ** 2))
    img[ys, xs] += g[..., None]


def _streak(img, rng, L, amp, tilt_deg):
    y, x = rng.uniform(0, H), rng.uniform(0, W)
    a = np.radians(tilt_deg)
    for s in np.linspace(0, L, int(L * 2) + 1):
        yy, xx = y + s * np.cos(a), x + s * np.sin(a)
        if 0 <= yy < H and 0 <= xx < W:
            img[int(yy), int(xx)] += amp


def add_precip(I, rng, rain, snow, hail):
    P = np.zeros_like(I)
    for _ in range(int(rain * 70)):
        _streak(P, rng, rng.uniform(6, 14), rng.uniform(0.08, 0.28), rng.normal(0, 8))
    for _ in range(int(snow * 45)):
        _splat(P, rng.uniform(0, H), rng.uniform(0, W), rng.uniform(0.9, 2.0), rng.uniform(0.25, 0.5))
    for _ in range(int(hail * 8)):
        y, x = rng.uniform(0, H), rng.uniform(0, W); s = rng.uniform(1.8, 3.0)
        for k in range(int(rng.integers(3, 6))):        # короткий смаз
            _splat(P, y + 3 * k, x + 0.8 * k, s, 0.35)
    return np.clip(I + gaussian_filter(P, (0.5, 0.5, 0)), 0, 1)


def lens_drop_mask(rng, n):
    m = np.zeros((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        y, x, s = rng.uniform(5, H - 5), rng.uniform(5, W - 5), rng.uniform(2.0, 4.5)
        m += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s ** 2))
    return np.clip(m, 0, 1)


def render_sequence(cfg: WxConfig):
    rng = np.random.default_rng(cfg.seed + 11)
    J = make_world(cfg.seed, cfg.terrain)
    d = depth_profile(cfg.alt, cfg.phi_top, cfg.phi_bot)[:, None, None]
    T = np.exp(-(3.912 / cfg.vis_m) * d)
    A = np.array([0.75, 0.76, 0.80])
    drop = lens_drop_mask(rng, cfg.drops) if cfg.drops else None
    fr = np.zeros((cfg.n, H, W, 3), np.float32)
    for t in range(cfg.n):
        ox = int(round(VX * t))
        I = J[:, ox:ox + W] * T + A * (1 - T)
        I = add_precip(I, rng, cfg.rain, cfg.snow, cfg.hail)
        if drop is not None:
            b = gaussian_filter(I, (3.5, 3.5, 0))
            I = I * (1 - drop[..., None]) + (0.8 * b + 0.04) * drop[..., None]
        if cfg.lens_fog > 0:
            f = cfg.lens_fog; b = gaussian_filter(I, (6 * f, 6 * f, 0)); c = 0.5 * f
            I = (1 - c) * b + c * b.mean()
        fr[t] = np.clip(I + rng.normal(0, cfg.noise, I.shape), 0, 1)
    return fr


# ---------------- LiDAR ----------------
@dataclass
class LidarConfig:
    seed: int = 0
    h: float = 10.0                 # высота сенсора над землёй, м
    d_az: float = 0.4               # угловое разрешение, град
    d_el: float = 0.4
    vis_m: float = 30000.0
    rain: float = 0.0
    snow: float = 0.0
    obstacles: int = 3
    cluster: float = 0.6            # доля помех со скоплениями (гладкое поле дальностей соседних лучей)


def lidar_scan(cfg: LidarConfig):
    """Возвращает dict: xyz (N,3), R, refl (калиброванная отражательная способность), clutter (bool), obstacle (bool)."""
    rng = np.random.default_rng(cfg.seed + 5)
    az = np.radians(np.arange(-30, 30, cfg.d_az)); el = np.radians(-np.arange(3, 25, cfg.d_el))
    AZ, EL = np.meshgrid(az, el)
    AZ, EL = AZ.ravel(), EL.ravel()
    R = cfg.h / np.sin(-EL)
    refl = np.exp(rng.normal(np.log(0.3), 0.4, R.size))
    obst = np.zeros(R.size, bool)
    # препятствия: узкий столб и «деревья» - угловые окна с меньшей дальностью
    for k in range(cfg.obstacles):
        a0 = rng.uniform(-25, 25); width = rng.choice([0.8, 2.0, 3.0]); r0 = rng.uniform(12, 90)
        m = (np.abs(np.degrees(AZ) - a0) < width / 2) & (np.degrees(EL) > -14 if width > 1 else True)
        R = np.where(m, np.minimum(R, r0), R); obst |= m
    clutter = np.zeros(R.size, bool)
    p = 1 - np.exp(-(0.35 * cfg.rain + 0.30 * cfg.snow))      # вероятность помехи на луч
    hit = rng.random(R.size) < p * 0.6
    zf = gaussian_filter(rng.normal(size=(len(el), len(az))), 2.0).ravel()
    zf = (zf - zf.min()) / (np.ptp(zf) + 1e-9)
    Rc = np.where(rng.random(R.size) < cfg.cluster, 2 + 23 * zf + rng.normal(0, 0.08, R.size), rng.uniform(2, 25, R.size))
    r_c = np.where(rng.random(R.size) < (cfg.snow / (cfg.rain + cfg.snow + 1e-9)),
                   np.exp(rng.normal(np.log(0.18), 0.5, R.size)), np.exp(rng.normal(np.log(0.05), 0.5, R.size)))
    R = np.where(hit, Rc, R); refl = np.where(hit, r_c, refl); clutter = hit; obst = obst & ~hit
    # затухание (дымка/облако): возврат теряется, если принятая мощность ниже порога
    bL = 3.912 / cfg.vis_m * (550 / 905) ** 1.3
    pw = refl * np.exp(-2 * bL * R) / R ** 2
    keep = pw > 2e-6
    x = R * np.cos(EL) * np.cos(AZ); y = R * np.cos(EL) * np.sin(AZ); z = R * np.sin(EL)
    return dict(xyz=np.c_[x, y, z][keep], R=R[keep], refl=refl[keep], clutter=clutter[keep], obstacle=obst[keep], n_rays=R.size)


# ---------------- IMU ----------------
FS = 200.0


def imu_series(seed=0, dur=20.0, hail_rate=0.0, turb=1.0, gusts=2, tone=0.0):
    """Вертикальное ускорение (м/с²), метки ударов. hail_rate - ударов в секунду; turb - масштаб турбулентности."""
    rng = np.random.default_rng(seed)
    n = int(dur * FS); t = np.arange(n) / FS
    low = gaussian_filter(rng.normal(0, 1, n), FS * 0.15) * 40 * turb          # низкочастотная турбулентность
    vib = rng.normal(0, 0.35, n)
    a = low + vib + tone * (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t))   # гармоники ротора
    for _ in range(gusts):                                                   # порывы: плавные «горбы»
        c, w, amp = rng.uniform(1, dur - 1), rng.uniform(0.3, 1.0), rng.uniform(3, 8) * turb
        a += amp * np.exp(-((t - c) ** 2) / (2 * (w / 2.5) ** 2))
    k = rng.poisson(hail_rate * dur); times = np.sort(rng.uniform(0.2, dur - 0.2, k)); amps = rng.uniform(2, 15, k)
    for tk, am in zip(times, amps):
        m = t >= tk
        a[m] += am * np.exp(-(t[m] - tk) / 0.03) * np.sin(2 * np.pi * 80 * (t[m] - tk)) * rng.choice([-1, 1])
    return t, a, times


def detect_impacts(a, thr_sigma=6.0, refractory=0.04, hp_hz=25.0):
    """Ударные события: фильтр верхних частот, порог в робастных σ (MAD), рефрактерный период."""
    sos = butter(4, hp_hz, btype="high", fs=FS, output="sos")
    h = sosfilt(sos, a)
    sig = 1.4826 * np.median(np.abs(h - np.median(h)))
    idx = np.where(np.abs(h) > thr_sigma * sig)[0]
    ev = []
    for i in idx:
        if not ev or (i - ev[-1]) / FS > refractory:
            ev.append(i)
    return np.array(ev) / FS, h, sig
