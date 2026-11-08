"""Синтетика оптических помех (угрозы 10 и 16): камера (автоэкспозиция, дробовой шум, блики) + тепловизор.

Модель кадра:
  L(y,x)  - радиометрическая яркость сцены (мир неподвижен, БВС смещается на VX px/кадр):
            ambient*текстура + города (точечные источники с ядром и ореолом) + костёр + блик солнца
  лазер   - подвижные пятна с мерцанием (импульсный режим), накладываются на кадр
  камера  - x = g*L, дробовой шум (Пуассон) + шум считывания, обрезка [0,1], гамма 1/2.2,
            автоэкспозиция g <- g*(target/mean(y))^beta
  тепловизор (радиометрический, К) - независим от видимого света: только фон и очаги огня.
Допущения: одноканальная яркость (цвет не моделируется), идеальная регистрация RGB/тепловизор.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from .synth_fog import make_texture

H = W = 96
FPS = 5.0
VX = 0.5                      # смещение сцены, px/кадр
FW = 250.0                    # ёмкость «лунки» (электроны) - масштаб дробового шума
READ_NOISE = 0.004
AE_TARGET, AE_BETA, G_MIN, G_MAX = 0.35, 0.15, 0.05, 40.0
AMBIENT_DAY, AMBIENT_NIGHT = 0.35, 0.006
WARMUP = 15


def psf_canvas(shape, srcs, halo=0.01, r0=3.0, p=2.2, core=0.7):
    """Сумма источников: гауссово ядро + степенной ореол. srcs: [(y, x, I)]."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    out = np.zeros(shape)
    for y, x, I in srcs:
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        out += I * (np.exp(-r2 / (2 * core ** 2)) + halo / (1 + (np.sqrt(r2) / r0) ** p))
    return out


@dataclass
class SeqConfig:
    n: int = 70
    night: bool = True
    urban: float = 0.0            # 0..1: плотность и яркость городской засветки (угроза 16)
    laser: bool = False
    laser_I: float = 1000.0       # пиковая яркость пятна (отн. единицы радиации)
    laser_core: float = 1.5       # радиус ядра пятна, px (sigma)
    laser_spots: int = 1
    laser_duty: float = 0.6
    laser_onset: int = 35
    laser_len: int = 30
    fire: bool = False
    fire_I: float = 60.0
    fire_dT: float = 300.0        # К
    fire_start: int = 30
    fire_ramp: int = 10
    glint: bool = False           # солнечный блик на воде (только днём), без тепловой аномалии
    glint_I: float = 8.0
    glint_start: int = 35
    noise_seed: int = -1          # зерно шума камеры/мерцания (-1: производное от seed); мир задаётся seed
    ae_mask_dilate: int = -1      # >=0: автоэкспозиция по пикселям вне насыщения, расширенного на k px (мера смягчения)
    ae_freeze_from: int = -1      # >=0: с этого кадра усиление заморожено (блокировка экспозиции после тревоги)
    halo: float = 0.01            # доля ореола в PSF (смена домена: 0.03)
    halo_r0: float = 3.0
    seed: int = 0


def make_sequence(cfg: SeqConfig):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    Wc = W + int(VX * n) + 6
    ambient = AMBIENT_NIGHT if cfg.night else AMBIENT_DAY

    # фон: текстура (мир), тепловой фон
    tex = make_texture(size=max(H, Wc), seed=cfg.seed)[:H, :Wc]
    base = ambient * (0.6 + 0.8 * tex)
    ttex = make_texture(size=max(H, Wc), seed=cfg.seed + 7)[:H, :Wc]
    Tworld = 285.0 + 4.0 * ttex

    # город
    city = np.zeros((H, Wc))
    if cfg.urban > 0:
        k = int(3 + 70 * cfg.urban)
        I = 10 ** rng.uniform(0.7, 1.3 + 0.7 * cfg.urban, k)
        srcs = [(rng.uniform(2, H - 2), rng.uniform(2, Wc - 2), i) for i in I]
        city = psf_canvas((H, Wc), srcs, halo=cfg.halo, r0=cfg.halo_r0)
    # костёр
    fire_map = np.zeros((H, Wc)); fire_T = np.zeros((H, Wc)); fy = fx = None
    if cfg.fire:
        fy, fx = rng.uniform(10, H - 10), rng.uniform(12, Wc - 12)
        fire_map = psf_canvas((H, Wc), [(fy, fx, cfg.fire_I)], halo=0.01, r0=2.0, core=1.6)
        yy, xx = np.mgrid[0:H, 0:Wc]
        fire_T = cfg.fire_dT * np.exp(-((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * 1.8 ** 2))
    # блик
    glint_map = np.zeros((H, Wc))
    if cfg.glint:
        gy, gx = rng.uniform(15, H - 15), rng.uniform(15, Wc - 15)
        glint_map = psf_canvas((H, Wc), [(gy, gx, cfg.glint_I)], halo=0.0, core=rng.uniform(3, 7))

    # лазер: траектории пятен в координатах кадра
    spots = []
    if cfg.laser:
        for _ in range(cfg.laser_spots):
            spots.append(dict(y=rng.uniform(15, H - 15), x=rng.uniform(15, W - 15),
                              vy=rng.normal(0, 0.8), vx=rng.normal(0, 0.8),
                              period=int(rng.integers(3, 9)), phase=int(rng.integers(0, 3))))
    t_l0, t_l1 = cfg.laser_onset, cfg.laser_onset + cfg.laser_len

    ys = np.zeros((n, H, W), np.float32); Ts = np.zeros((n, H, W), np.float32)
    gain = np.zeros(n); lab = {k: np.zeros(n, bool) for k in ("laser", "laser_pulse", "fire", "glint")}
    # начальное усиление: стационарное решение автоэкспозиции для кадра без событий
    L0 = base[:, :W] + city[:, :W]
    g = 1.0
    for _ in range(60):
        y0 = np.clip(g * L0, 0, 1) ** (1 / 2.2)
        g = float(np.clip(g * (AE_TARGET / max(y0.mean(), 1e-3)) ** 0.5, G_MIN, G_MAX))
    fire_flick = 1.0
    rc = np.random.default_rng(cfg.seed + 777 if cfg.noise_seed < 0 else cfg.noise_seed)
    for t in range(n):
        ox = int(round(VX * t))
        L = base[:, ox:ox + W] + city[:, ox:ox + W]
        if cfg.fire and t >= cfg.fire_start:
            fire_flick = np.clip(0.9 * fire_flick + 0.1 + rc.normal(0, 0.06), 0.6, 1.4)
            ramp = min(1.0, (t - cfg.fire_start + 1) / cfg.fire_ramp)
            L = L + ramp * fire_flick * fire_map[:, ox:ox + W]
            lab["fire"][t] = True
        if cfg.glint and t >= cfg.glint_start:
            ramp = min(1.0, (t - cfg.glint_start + 1) / 3)
            L = L + ramp * glint_map[:, ox:ox + W]
            lab["glint"][t] = True
        if cfg.laser and t_l0 <= t < t_l1:
            lab["laser"][t] = True
            srcs = []
            for s in spots:
                s["y"] = np.clip(s["y"] + s["vy"], 5, H - 5); s["x"] = np.clip(s["x"] + s["vx"], 5, W - 5)
                on = ((t + s["phase"]) % s["period"]) < max(1, int(round(cfg.laser_duty * s["period"])))
                if on:
                    srcs.append((s["y"], s["x"], cfg.laser_I))
            if srcs:
                lab["laser_pulse"][t] = True
                L = L + psf_canvas((H, W), srcs, halo=0.003, r0=4.0, core=cfg.laser_core)
        # камера
        x = g * L
        e = rc.poisson(np.clip(x, 0, 1.5) * FW) / FW
        x = np.clip(e + rc.normal(0, READ_NOISE, x.shape), 0, 1)
        y = x ** (1 / 2.2)
        gain[t] = g
        if cfg.ae_mask_dilate >= 0:
            m = y >= 0.98
            if cfg.ae_mask_dilate > 0:
                m = binary_dilation(m, iterations=cfg.ae_mask_dilate)
            ym = y[~m].mean() if (~m).any() else y.mean()
        else:
            ym = y.mean()
        if not (cfg.ae_freeze_from >= 0 and t + 1 >= cfg.ae_freeze_from):
            g = float(np.clip(g * (AE_TARGET / max(ym, 1e-3)) ** AE_BETA, G_MIN, G_MAX))
        ys[t] = y
        T = Tworld[:, ox:ox + W] + rc.normal(0, 0.15, (H, W))
        if cfg.fire and t >= cfg.fire_start:
            T = T + ramp * fire_flick * fire_T[:, ox:ox + W]
        Ts[t] = T
    return dict(y=ys, T=Ts, gain=gain, labels=lab, cfg=cfg)
