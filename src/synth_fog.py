"""Синтетический сценарий запотевания оптики (угроза 28).

Физика:
  * воздух: T(alt), Td(alt) из стандартной атмосферы (physics.air_at_altitude);
  * линза: dT_lens/dt = (T_air + offset - T_lens)/tau
    (offset < 0: радиационное выстывание/обдув, offset > 0: нагрев электроникой);
  * конденсат: f' = gain*max(Td - T_lens, 0)*(1 - f) - decay*f*[Td <= T_lens];
  * видео: гауссово размытие + потеря контраста, зависящие от f;
  * датчики БВС-5: T и RH с шумом и инерцией датчика.
Истинное состояние (f, T_lens) в «телеметрию» не попадает: детекторы видят только измерения.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace

from .physics import air_at_altitude, dewpoint, rel_humidity
from .detectors import lag_filter

SIGMA_MAX = 6.0      # макс. размытие, px
CONTRAST_LOSS = 0.5  # макс. потеря контраста при f=1


@dataclass
class FogScenario:
    dt: float = 1.0
    duration: float = 900.0
    cruise_alt: float = 1000.0
    descent_start: float = 300.0
    descent_rate: float = 5.0
    ground_alt: float = 30.0
    ground_T: float = 22.0
    ground_Td: float = 18.0
    lapse_Td: float = 0.006     # К/м: падение Td с высотой (сухой слой выше)
    tau: float = 120.0          # постоянная времени линзы, с
    lens_offset: float = -1.5   # К
    sensor_tau: float = 10.0    # инерция датчика T/RH, с
    sigma_T: float = 0.3        # К
    sigma_RH: float = 2.0       # %
    fog_gain: float = 0.02      # 1/(с*К)
    fog_decay: float = 0.01     # 1/с
    seed: int = 0


# --- видео ---------------------------------------------------------------

def make_texture(size=128, seed=0):
    """Синтетическая «местность»: многомасштабный шум + прямоугольники (контуры)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    for s, w in [(1, 0.5), (3, 1.0), (8, 1.5)]:
        img += w * gaussian_filter(rng.normal(size=(size, size)), s)
    for _ in range(12):
        x, y = rng.integers(0, size - 20, 2)
        w, h = rng.integers(5, 20, 2)
        img[y:y + h, x:x + w] += rng.normal(0, 1.5)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    return img


def render_frame(img, f):
    """Применить запотевание f∈[0,1] к кадру."""
    out = gaussian_filter(img, SIGMA_MAX * f) if f > 0 else img.copy()
    c = CONTRAST_LOSS * f
    return (1 - c) * out + c * out.mean()


def lap_var(img):
    """Резкость: дисперсия лапласиана."""
    return float(np.var(laplace(img)))


def sharpness_lut(img, n=41):
    """Таблица резкость(f) для быстрой генерации длинных рядов."""
    fs = np.linspace(0, 1, n)
    s = np.array([lap_var(render_frame(img, f)) for f in fs])
    return fs, s / s[0]


# --- симуляция ---------------------------------------------------------------

def simulate(sc: FogScenario, lut=None):
    """Возвращает DataFrame: истина (t_lens, fog, label) + измерения (T_meas, RH_meas, sharp)."""
    rng = np.random.default_rng(sc.seed)
    n = int(sc.duration / sc.dt)
    t = np.arange(n) * sc.dt

    # профиль высоты: крейсер -> спуск -> земля
    alt = np.full(n, sc.cruise_alt)
    d = t >= sc.descent_start
    alt[d] = np.maximum(sc.cruise_alt - sc.descent_rate * (t[d] - sc.descent_start), sc.ground_alt)

    T_air, Td_air = air_at_altitude(sc.ground_T, sc.ground_Td, alt - sc.ground_alt,
                                     lapse_Td=sc.lapse_Td)

    # линза стартует в равновесии с крейсерским состоянием
    T_target = T_air + sc.lens_offset
    T_lens = lag_filter(T_target, sc.tau, sc.dt, x0=T_target[0])

    # конденсат
    drive = Td_air - T_lens
    fog = np.zeros(n)
    for i in range(1, n):
        grow = sc.fog_gain * max(drive[i], 0.0) * (1 - fog[i - 1])
        dec = sc.fog_decay * fog[i - 1] if drive[i] <= 0 else 0.0
        fog[i] = np.clip(fog[i - 1] + sc.dt * (grow - dec), 0, 1)
    label = fog > 0.15

    # измерения БВС-5
    T_s = lag_filter(T_air, sc.sensor_tau, sc.dt, x0=T_air[0])
    RH_true = rel_humidity(T_air, Td_air)
    RH_s = lag_filter(RH_true, sc.sensor_tau, sc.dt, x0=RH_true[0])
    T_meas = T_s + rng.normal(0, sc.sigma_T, n)
    RH_meas = np.clip(RH_s + rng.normal(0, sc.sigma_RH, n), 1, 100)

    # видео: сцена меняется (AR(1) в лог-шкале) * эффект запотевания * шум измерения
    if lut is None:
        lut = sharpness_lut(make_texture(seed=sc.seed))
    fs, s_ratio = lut
    scene = np.zeros(n)
    for i in range(1, n):
        scene[i] = 0.98 * scene[i - 1] + rng.normal(0, 0.03)
    sharp = np.exp(scene) * np.interp(fog, fs, s_ratio) * np.exp(rng.normal(0, 0.03, n))

    return pd.DataFrame(dict(
        t=t, alt=alt, T_air=T_air, Td_air=Td_air, T_lens=T_lens, fog=fog, label=label,
        T_meas=T_meas, RH_meas=RH_meas, Td_meas=dewpoint(T_meas, RH_meas), sharp=sharp))


def random_scenario(rng, seed):
    """Случайный сценарий. На крейсере тумана нет (отбраковка), на снижении он может возникнуть."""
    while True:
        ground_T = rng.uniform(5, 30)
        spread = rng.uniform(1.0, 7.0)           # T - Td у земли
        sc = FogScenario(
            ground_T=ground_T, ground_Td=ground_T - spread,
            lapse_Td=rng.uniform(0.003, 0.009),
            cruise_alt=rng.uniform(300, 1500),
            tau=rng.uniform(40, 300),
            lens_offset=rng.uniform(-3.0, 1.0),
            descent_rate=rng.uniform(3, 8),
            seed=seed)
        dz = sc.cruise_alt - sc.ground_alt
        T_c, Td_c = air_at_altitude(sc.ground_T, sc.ground_Td, dz, lapse_Td=sc.lapse_Td)
        if Td_c < T_c + sc.lens_offset - 0.5:      # линза на крейсере суше точки росы
            return sc
