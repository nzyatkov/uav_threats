"""Генератор потоков данных по «Потоки_данных.pdf» (разделы 5.2, 5.3) + сетевая модель + шина.

Эмулируются только поля, перечисленные в PDF: БВС-1 (телеметрия), БВС-2 (ГНСС), БВС-4 (препятствия: lidar/radar/camera),
БВС-5 (метео и энергия), БВС-7 (диагностика), НСУ-2 (зоны), НСУ-3 (метеопрогноз). Видеопотока и сырого IMU нет.
Скрытый «истинный» процесс атмосферы/платформы (сетка 10 Гц) порождает значения полей; детекторам видны только сообщения
с их частотами, задержками и потерями. Ключевая особенность БВС-4: сообщения идут ТОЛЬКО при обнаружении (в штатном режиме 0 Гц).
"""
from dataclasses import dataclass, field
import numpy as np
from .physics import air_at_altitude, rel_humidity, wet_bulb_stull
from .detectors import lag_filter

DT = 0.1
UAV = "UAV-001"
LAT0, LON0 = 50.11, 8.68
R_EARTH = 6371000.0

# --- спецификация из PDF (для валидации генератора и бюджетов) ---
SPEC = {
    "BVS1": dict(rate=(5, 10), max_latency=0.5, loss="best_effort", size=256),
    "BVS2": dict(rate=(1, 2), max_latency=1.0, loss="guaranteed", size=128),
    "BVS3": dict(rate="event", max_latency=0.2, loss="guaranteed", size=512),
    "BVS4": dict(rate=(5, 20), max_latency=0.3, loss="metadata guaranteed", size=600),   # 1-50 КБ с облаком точек
    "BVS5": dict(rate=(1, 1), max_latency=1.0, loss="best_effort", size=192),
    "BVS7": dict(rate=(0.1, 0.2), max_latency=2.0, loss="errors guaranteed", size=384),
    "NSU2": dict(rate="event+heartbeat 300 s", max_latency=1.0, loss="critical guaranteed", size=400),
    "NSU3": dict(rate=(0.1, 0.2), max_latency=2.0, loss="best_effort", size=450),
}
# сетевая модель: медиана и p99 задержки (с), вероятность потери (допущение внутри требований PDF)
NET = {"BVS1": (0.06, 0.40, 0.02), "BVS2": (0.12, 0.90, 0.0), "BVS3": (0.03, 0.15, 0.0), "BVS4": (0.08, 0.28, 0.0),
       "BVS5": (0.15, 0.90, 0.03), "BVS7": (0.25, 1.80, 0.02), "NSU2": (0.20, 0.90, 0.0), "NSU3": (0.30, 1.80, 0.05)}

# параметры модели сенсоров/прогноза (допущения; меняются в экспериментах на смену условий)
LIDAR_K = 0.15        # дальность lidar = min(150, LIDAR_K * V)
CAM_K = 0.4           # дальность камеры = min(120, CAM_K * V) * ...
NSU_SIGMA = 0.6       # σ ошибки внешнего прогноза видимости в лог-шкале

EVENT_KINDS = ["rain", "snow", "hail", "fog_cloud", "dust", "smoke", "laser"]
TRUTH_KINDS = EVENT_KINDS + ["lens_fog", "night_city"]


@dataclass
class Ev:
    kind: str
    t0: float
    t1: float
    I: float = 1.0


@dataclass
class ScenarioConfig:
    kind: str = "clear"
    seed: int = 0
    duration: float = 600.0
    T0: float = 18.0
    Td0: float = 10.0
    lapse_Td: float = 0.002
    V0: float = 30000.0
    wind0: float = 4.0
    night: bool = False
    urban: float = 0.0
    cruise_alt: float = 120.0
    descent_start: float = 1e9
    descent_rate: float = 5.0
    lens_tau: float = 120.0
    lens_offset: float = -1.0
    events: list = field(default_factory=list)
    zone_announced: bool = False
    obstacle_rate: float = 1 / 45
    hum_coupling: float = 1.0       # сила влияния событий на T/T_d (0 - события не меняют влажность; для проверки смены условий)
    speed: float = 12.0


def make_scenario(kind, seed, duration=None):
    r = np.random.default_rng(seed + 7)
    c = ScenarioConfig(kind=kind, seed=seed)
    c.T0 = r.uniform(5, 25); c.Td0 = c.T0 - r.uniform(5, 12); c.wind0 = r.uniform(2, 7)
    c.duration = duration or 600.0
    t0 = r.uniform(120, 250)
    if kind == "humid":
        c.Td0 = c.T0 - r.uniform(0.5, 2.0)
    elif kind == "rain":
        c.T0 = r.uniform(5, 22); c.Td0 = c.T0 - 2.5; c.events = [Ev("rain", t0, t0 + r.uniform(120, 250), r.uniform(.3, 1))]
    elif kind == "snow":
        c.T0 = r.uniform(-5, 1); c.Td0 = c.T0 - 3.0; c.events = [Ev("snow", t0, t0 + r.uniform(120, 250), r.uniform(.3, 1))]
    elif kind == "hail":
        c.T0 = r.uniform(14, 26); c.Td0 = c.T0 - 4; I = r.uniform(.3, 1); L = r.uniform(30, 90)
        c.events = [Ev("rain", t0 - 20, t0 + L + 20, .5), Ev("hail", t0, t0 + L, I)]; c.wind0 = r.uniform(5, 10)
    elif kind == "fog_cloud":
        c.Td0 = c.T0 - 1.5; c.events = [Ev("fog_cloud", t0, t0 + r.uniform(100, 200), r.uniform(.7, 1))]
    elif kind == "dust":
        c.T0 = r.uniform(20, 35); c.Td0 = c.T0 - r.uniform(12, 20); c.wind0 = r.uniform(8, 14)
        c.events = [Ev("dust", t0, t0 + r.uniform(150, 300), r.uniform(.4, 1))]
    elif kind == "smoke":
        c.T0 = r.uniform(20, 32); c.Td0 = c.T0 - 8; c.events = [Ev("smoke", t0, t0 + r.uniform(150, 300), r.uniform(.4, 1))]
    elif kind == "lens_fog":
        c.T0 = r.uniform(16, 26); c.Td0 = c.T0 - 4; c.lapse_Td = 0.006; c.cruise_alt = 1000.0
        c.descent_start = 300.0; c.duration = duration or 900.0; c.lens_tau = r.uniform(90, 220); c.lens_offset = r.uniform(-2.0, -0.5)
    elif kind == "laser":
        c.night = True; c.urban = r.uniform(.2, .7); c.events = [Ev("laser", t0, t0 + r.uniform(20, 40), 1.0)]
        c.zone_announced = bool(r.random() < 0.5)
    elif kind == "night_city":
        c.night = True; c.urban = r.uniform(.4, 1.0)
    return c


def _ramp(t, ev, ramp=1.0):
    return ev.I * np.clip((t - ev.t0) / ramp, 0, 1) * np.clip((ev.t1 - t) / ramp, 0, 1)


def compute_state(cfg: ScenarioConfig):
    """Скрытое состояние на сетке 10 Гц."""
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(0, cfg.duration, DT); n = len(t)
    ground = 30.0
    alt = np.interp(t, [0, 20, cfg.descent_start, cfg.descent_start + (cfg.cruise_alt - ground) / cfg.descent_rate, 1e9],
                    [ground, cfg.cruise_alt, cfg.cruise_alt, ground, ground])
    I = {k: np.zeros(n) for k in EVENT_KINDS}
    for e in cfg.events:
        I[e.kind] = np.maximum(I[e.kind], _ramp(t, e))
    Tair, Td = air_at_altitude(cfg.T0, cfg.Td0, alt - ground, lapse_Td=cfg.lapse_Td)
    hc = cfg.hum_coupling
    Ih = {k: v * hc for k, v in I.items()}            # эффективная интенсивность воздействия на температуру/влажность
    Tair = Tair - 1.0 * Ih["rain"]
    spread = np.maximum(Tair - Td, 0.05)
    spread = np.where(Ih["rain"] + Ih["snow"] > 0, np.maximum(0.3, spread * (1 - 0.85 * np.maximum(Ih["rain"], Ih["snow"]))), spread)
    spread = np.where(Ih["fog_cloud"] > 0, 0.05 + (spread - 0.05) * (1 - Ih["fog_cloud"]), spread)
    spread = spread + 6.0 * Ih["dust"] + 2.0 * Ih["smoke"]
    Td = Tair - spread
    RH = np.clip(rel_humidity(Tair, Td), 1, 100)
    # линза
    T_lens = lag_filter(Tair + cfg.lens_offset, cfg.lens_tau, DT, x0=(Tair + cfg.lens_offset)[0])
    drive = Td - T_lens; f = np.zeros(n)
    for i in range(1, n):
        f[i] = np.clip(f[i - 1] + DT * (0.02 * max(drive[i], 0) * (1 - f[i - 1]) - (0.01 * f[i - 1] if drive[i] <= 0 else 0)), 0, 1)
    # видимость
    inv = 1 / cfg.V0 + 1 / (60 + 300 * (1 - I["fog_cloud"]) + 1e9 * (I["fog_cloud"] == 0)) \
        + 1 / (300 + 3700 * (1 - I["dust"]) + 1e9 * (I["dust"] == 0)) + 1 / (500 + 7500 * (1 - I["smoke"]) + 1e9 * (I["smoke"] == 0)) \
        + 1 / (cfg.V0 * (1 - 0.7 * I["rain"] + 1e-3) * 0.5 + 1e9 * (I["rain"] == 0)) + 1 / (4000 * (1 - 0.85 * I["snow"] + 1e-3) + 1e9 * (I["snow"] == 0))
    V = 1 / inv
    # ветер, турбулентность
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.97 * ou[i - 1] + rng.normal(0, 0.25)
    gain = 1 + 2.5 * I["hail"] + 1.5 * I["dust"]
    wind = np.clip(cfg.wind0 + (0.5 + 0.35 * gain) * ou, 0, None); gust = wind + np.abs(rng.normal(0, 0.6 * gain)) + 1.5 * I["hail"]
    turb = np.clip(0.1 + 0.3 * wind / 10 + 0.5 * I["hail"] + 0.4 * I["dust"] + rng.normal(0, 0.03, n), 0, 1)
    speed = cfg.speed + rng.normal(0, 0.4, n)
    # энергия
    acc = np.cumsum(0.0008 * I["snow"] * DT); delta = np.minimum(acc, 0.25) + 0.03 * I["rain"] + 0.01 * I["dust"]
    P = (120 + 0.55 * speed ** 2 + 2.5 * gust) * (1 + delta) * (1 + rng.normal(0, 0.02, n))
    E_tot = 90 * 3600.0; batt = 100 - np.cumsum(P * DT) / E_tot * 100
    # ориентация
    sig_att = 0.4 + 1.2 * turb
    pitch = rng.normal(0, sig_att); roll = rng.normal(0, sig_att)
    for e in cfg.events:
        if e.kind == "hail":
            k = rng.poisson(4 * e.I * (e.t1 - e.t0)); tt = rng.uniform(e.t0, e.t1, k)
            for x in tt:
                i = int(x / DT)
                if i < n - 2:
                    a = rng.uniform(3, 8) * rng.choice([-1, 1]); pitch[i:i + 2] += a; roll[i:i + 2] += -0.6 * a
    dev = np.abs(gaussian_ar(rng, n, 0.98) * (0.5 + 1.5 * (gust - wind) + 0.3 * turb * 5))
    cum_dust = np.cumsum(I["dust"] * DT)
    cum_hail = np.cumsum(I["hail"] * DT)
    laser_on = I["laser"] > 0.5
    return dict(t=t, alt=alt, T=Tair, Td=Td, RH=RH, T_lens=T_lens, lens_f=f, V=V, wind=wind, gust=gust, turb=turb, speed=speed,
                P=P, delta=delta, batt=batt, pitch=pitch, roll=roll, dev=dev, I=I, laser_on=laser_on, cum_dust=cum_dust, cum_hail=cum_hail,
                night_city=np.full(n, cfg.night * cfg.urban), E_tot=E_tot)


def gaussian_ar(rng, n, phi):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    return x


def truth_labels(cfg, S):
    tr = {k: S["I"][k] > 0.2 for k in EVENT_KINDS}
    tr["lens_fog"] = S["lens_f"] > 0.15
    tr["night_city"] = S["night_city"] > 0.3
    return tr


def _latlon(cfg, t):
    d = cfg.speed * t
    return LAT0 + np.degrees(d * 0.0 / R_EARTH), LON0 + np.degrees(d / (R_EARTH * np.cos(np.radians(LAT0))))


def _gen_bvs1(cfg, S, rng):
    out = []; step = 0.125                    # 8 Гц, внутри 5-10 Гц
    for tt in np.arange(0, cfg.duration, step):
        i = int(tt / DT); lat, lon = _latlon(cfg, tt)
        out.append(("BVS1", tt, dict(timestamp=int(tt * 1000), uav_id=UAV, lat=float(lat), lon=float(lon), alt=float(S["alt"][i]), speed=float(S["speed"][i]),
                                     course=90.0, pitch=float(S["pitch"][i]), roll=float(S["roll"][i]), yaw=90.0 + float(rng.normal(0, 1)),
                                     battery_percent=float(S["batt"][i]), mission_status="cruise", nav_quality_flag=1)))
    return out


def _gen_bvs2(cfg, S, rng):
    return [("BVS2", tt, dict(gps_fix_type=3, satellites_count=int(rng.integers(14, 19)), hdop=float(rng.normal(0.9, 0.05)), vdop=1.2,
                              gnss_signal_strength=-130 + float(rng.normal(0, 1)), ins_drift_estimate=0.0, nav_solution_status="ok", data_validity_flag=1))
            for tt in np.arange(0.5, cfg.duration, 0.75)]


def _gen_bvs5(cfg, S, rng):
    out = []
    for tt in np.arange(0, cfg.duration, 1.0):
        i = int(tt / DT)
        out.append(("BVS5", tt, dict(timestamp=int(tt * 1000), wind_speed=float(S["wind"][i] + rng.normal(0, 0.5)), wind_direction=float(rng.normal(270, 10)),
                                     wind_gust=float(S["gust"][i] + rng.normal(0, 0.5)), temperature=float(S["T"][i] + rng.normal(0, 0.3)),
                                     humidity=float(np.clip(S["RH"][i] + rng.normal(0, 2.0), 1, 100)), power_consumption=float(S["P"][i]),
                                     estimated_flight_time=float(S["batt"][i] / 100 * S["E_tot"] / S["P"][i]), deviation_from_route=float(S["dev"][i]),
                                     stabilization_status="ok" if abs(S["pitch"][i]) < 3 else "degraded", estimate_reliability_flag=1)))
    return out


def _gen_bvs7(cfg, S, rng):
    out = []; n = len(S["t"])
    for tt in np.arange(2.0, cfg.duration, 7.0):                    # ~0.14 Гц
        i = int(tt / DT)
        esc = "warn" if S["cum_dust"][i] > 40 else "ok"
        imu = "degraded" if S["cum_hail"][i] > 25 else "ok"
        out.append(("BVS7", tt, dict(system_health=dict(cpu_temp=float(52 + 0.4 * (S["T"][i] - 20) + 14 * S["I"]["dust"][i] + rng.normal(0, 0.8)),
                                                         mem_usage=float(45 + rng.normal(0, 2)), disk_free=int(8e9 - tt * 1e5)),
                                     component_status=dict(motors="ok", ESC=esc, IMU=imu, compass="ok", baro="ok"), error_codes=[], warning_flags=[],
                                     uptime=float(tt), last_reboot_reason="none", data_freshness_flag=1, component_id="SYS")))
    # событийные ошибки: удары града -> IMU, пыль -> ESC перегрев
    for e in cfg.events:
        if e.kind == "hail":
            for x in np.sort(rng.uniform(e.t0, e.t1, rng.poisson(1.2 * e.I * (e.t1 - e.t0)))):
                out.append(("BVS7", x, dict(system_health={}, component_status={}, error_codes=[0x2101], warning_flags=["IMU_SHOCK"], uptime=float(x),
                                            last_reboot_reason="none", data_freshness_flag=1, component_id="IMU")))
        if e.kind == "dust":
            x = e.t0 + 60 / max(e.I, .3)
            if x < e.t1:
                out.append(("BVS7", x, dict(system_health={}, component_status=dict(ESC="warn"), error_codes=[0x3105], warning_flags=["ESC_OVERTEMP"],
                                            uptime=float(x), last_reboot_reason="none", data_freshness_flag=1, component_id="ESC")))
    return out


def _gen_nsu3(cfg, S, rng):
    out = []
    for k, tt in enumerate(np.arange(1.0, cfg.duration, 6.0)):      # ~0.17 Гц
        i = int(tt / DT)
        v_true = S["V"][i]; v_f = float(np.exp(0.5 * np.log(v_true) + 0.5 * np.log(cfg.V0) + rng.normal(0, NSU_SIGMA)))
        pr = float(np.clip(0.1 * rng.random() + 0.75 * float(max(S["I"]["rain"][i], S["I"]["snow"][i], S["I"]["hail"][i]) > 0) + rng.normal(0, 0.1), 0, 1))
        out.append(("NSU3", tt, dict(forecast_id=f"fc-{k}", turbulence_index=float(np.clip(S["turb"][i] + rng.normal(0, 0.1), 0, 1)),
                                     visibility_m=v_f, precipitation_prob=pr, valid_from=int(tt * 1000), valid_to=int((tt + 60) * 1000),
                                     model_source="synthetic", forecast_confidence=float(rng.uniform(.5, .9)),
                                     wind_profile=dict(speed_mps=[float(S["wind"][i])], gust_mps=[float(S["gust"][i])], altitude_m=[float(S["alt"][i])]))))
    return out


def _gen_nsu2(cfg, S, rng):
    out = []
    for e in cfg.events:
        if e.kind == "laser" and cfg.zone_announced:
            lat, lon = _latlon(cfg, 0.5 * (e.t0 + e.t1))
            out.append(("NSU2", 3.0, dict(update_id="z-laser", update_type="no_fly_zone", geometry=dict(type="circle", lat=float(lat), lon=float(lon), radius_m=900.0,
                                                                                                        altitude_min=0.0, altitude_max=300.0),
                                           validity_window=dict(start_ts=int((e.t0 - 30) * 1000), end_ts=int((e.t1 + 30) * 1000)),
                                           source_authority="regulator", confidence_level=0.9, geometry_validation_hash="x")))
    return out


def _gen_bvs4(cfg, S, rng):
    """Сообщения БВС-4 только при обнаружении. Реальные препятствия + фантомы (помехи погоды/света)."""
    out = []; n = len(S["t"])
    # реальные препятствия: подход 150 -> 20 м за 8 с
    obs = []; tcur = 10.0
    while tcur < cfg.duration - 10:
        tcur += rng.exponential(1 / cfg.obstacle_rate); obs.append((tcur, rng.uniform(-30, 30), rng.uniform(-5, 10), rng.uniform(1, 15)))
    for step in range(0, n, 1):                                       # скан 10 Гц
        i = step; tt = S["t"][i]
        V = S["V"][i]; P_ = max(S["I"]["rain"][i], S["I"]["snow"][i]); dust = S["I"]["dust"][i]; smoke = S["I"]["smoke"][i]
        lf = S["lens_f"][i]; las = float(S["laser_on"][i]); nc = S["night_city"][i]
        Rmax = {"lidar": min(150.0, LIDAR_K * V), "radar": 200.0, "camera": min(120.0, CAM_K * V) * (1 - 0.9 * lf) * (0.15 if las else 1.0)}   # допущение: дальность падает с V
        pdet = {"lidar": 0.97, "radar": 0.97, "camera": 0.9 * (1 - 0.7 * lf) * (1 - 0.9 * las)}
        rate_ph = {"lidar": 0.01 + 2.0 * (S["I"]["rain"][i] + S["I"]["snow"][i]) + 1.5 * dust + 0.8 * smoke,
                   "radar": 0.005 + 0.15 * S["I"]["rain"][i] + 0.1 * S["I"]["hail"][i],
                   "camera": 0.01 + 0.35 * nc + 0.25 * las + 0.08 * P_}
        health = {"lidar": "DEGRADED" if (P_ > 0.5 or V < 1000 or dust > 0.5) else "OK", "radar": "OK",
                  "camera": "FAIL" if las and rng.random() < .5 else "DEGRADED" if (lf > 0.3 or V < 500) else "OK"}
        # истинные препятствия
        true_d = None
        for (t_o, brg, el, sz) in obs:
            if 0 <= tt - t_o <= 8:
                d = 150 - 130 * (tt - t_o) / 8; true_d = (d, brg, el, sz)
        for s in ("lidar", "radar", "camera"):
            if true_d is not None and true_d[0] <= Rmax[s] and rng.random() < pdet[s]:
                d, brg, el, sz = true_d
                tv = np.exp(-3.912 * d / V)
                conf = {"lidar": 0.92, "radar": 0.9, "camera": 0.9 * np.sqrt(tv) * (1 - 0.8 * lf) * (1 - 0.9 * las)}[s] + rng.normal(0, 0.05)
                out.append(("BVS4", tt, _b4(s, d + rng.normal(0, 0.05), brg + rng.normal(0, .5), el, sz, conf, health[s], tt, rng, False)))
            if rng.random() < rate_ph[s] * DT:
                d = rng.uniform(2, 15) if s == "lidar" else rng.uniform(5, 80)
                sz = rng.uniform(0.05, 0.4) if s != "radar" else rng.uniform(0.3, 2.0)
                out.append(("BVS4", tt, _b4(s, d, rng.uniform(-30, 30), rng.uniform(-10, 10), sz, float(np.clip(rng.beta(2, 4), 0.05, 0.9)), health[s], tt, rng, True)))
    return out


def _b4(s, d, brg, el, sz, conf, health, tt, rng, phantom):
    return dict(sensor_type=s, obstacle_detected=True, distance=float(d), bearing=float(brg), elevation=float(el), size_estimate=float(sz),
                point_cloud_ref=(f"pc-{int(tt * 1000)}" if (s == "lidar" and rng.random() < 0.1) else None), timestamp=int(tt * 1000),
                detection_confidence=float(np.clip(conf, 0, 1)), sensor_health_status=health, sensor_serial=f"{s}-01", _phantom=phantom)


def apply_network(raw, rng):
    msgs = []; seq = {}
    for stream, t_gen, p in sorted(raw, key=lambda x: x[1]):
        med, p99, loss = NET[stream]
        if rng.random() < loss:
            continue
        lat = float(np.exp(rng.normal(np.log(med), np.log(p99 / med) / 2.326)))
        seq[stream] = seq.get(stream, 0) + 1
        msgs.append(dict(stream=stream, seq=seq[stream], t_gen=float(t_gen), t_arrive=float(t_gen + lat), size=SPEC[stream]["size"], p=p))
    msgs.sort(key=lambda m: m["t_arrive"])
    return msgs


def generate_bus(cfg: ScenarioConfig, skip=()):
    """Возвращает (сообщения в порядке прибытия, истина по меткам, скрытое состояние). skip - потоки, которые не генерировать
    (для ускорения; генераторы потоков используют независимые подпотоки случайности, поэтому содержимое остальных потоков не меняется)."""
    S = compute_state(cfg)
    gens = [("BVS1", _gen_bvs1), ("BVS2", _gen_bvs2), ("BVS5", _gen_bvs5), ("BVS7", _gen_bvs7), ("NSU3", _gen_nsu3), ("NSU2", _gen_nsu2), ("BVS4", _gen_bvs4)]
    raw = []
    for k, (name, g) in enumerate(gens):
        r = np.random.default_rng(cfg.seed * 10 + 1000 + k)      # независимый подпоток на каждый поток
        out = g(cfg, S, r)
        if name not in skip:
            raw += out
    rng = np.random.default_rng(cfg.seed + 999)
    return apply_network(raw, rng), truth_labels(cfg, S), S
