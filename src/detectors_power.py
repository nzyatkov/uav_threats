"""Детекторы по мощности и устойчивости: остаток мощности (налипание/намокание), град (attitude + ошибки IMU + прогноз), перегрев (cpu_temp)."""
import math
from collections import deque
import numpy as np
from .harness import Detector, _alert
from .alerts import Gate


def fit_power_model(cache, kinds=("clear", "humid", "laser", "night_city"), tau_g=10.0):
    """Калибровка ожидаемой мощности P_exp = θ0 + θ1 v² + θ2 g_s по безопасным полётам (ridge); g_s - порыв, сглаженный EWMA (шум измерения)."""
    X, y = [], []
    for c in cache:
        if c["kind"] not in kinds: continue
        v = None; gs = None; tl = None
        for m in c["msgs"]:
            if m["stream"] == "BVS1": v = m["p"]["speed"]
            elif m["stream"] == "BVS5" and v is not None:
                p = m["p"]; t = p["timestamp"] / 1000.0
                if gs is None: gs = p["wind_gust"]
                else: gs += (1 - math.exp(-(t - tl) / tau_g)) * (p["wind_gust"] - gs)
                tl = t
                if t > 25.0: X.append([1.0, v * v, gs]); y.append(p["power_consumption"])
    X = np.array(X); y = np.array(y); lam = 1e-3
    th = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
    return th, float(np.std(y / (X @ th) - 1))


def fit_cpu_model(cache, kinds=("clear", "humid", "laser", "night_city", "rain", "snow")):
    """Ожидаемая cpu_temp = c0 + c1 (T − 20) по полётам без пыли. Возвращает (c0, c1, σ)."""
    T = None; X, y = [], []
    for c in cache:
        if c["kind"] not in kinds: continue
        for m in c["msgs"]:
            if m["stream"] == "BVS5": T = m["p"]["temperature"]
            elif m["stream"] == "BVS7" and m["p"]["system_health"] and T is not None:
                X.append([1.0, T - 20.0]); y.append(m["p"]["system_health"]["cpu_temp"])
    X = np.array(X); y = np.array(y); th = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(th[0]), float(th[1]), float(np.std(y - X @ th))


class PowerResidualDetector(Detector):
    """Остаток r = P/P_exp − 1 (БВС-5, скорость из БВС-1). Режимы: 'ewma' - Gate по сглаженной прибавке δ̂ (порог on/off, n подряд, пауза);
    'cusum' - CUSUM вверх с допуском k и порогом h (сравнение)."""
    name = "power_residual"; streams = ("BVS1", "BVS5")

    def __init__(self, theta, tau=60.0, mode="ewma", on=0.02, off=0.01, n_on=3, n_off=10, cooldown=0.0, k=0.02, h=0.3, tau_g=10.0, warmup=25.0):
        self.th, self.tau, self.mode, self.k, self.h, self.tau_g, self.warmup = theta, tau, mode, k, h, tau_g, warmup
        self.gate = Gate(on, off, n_on, n_off, cooldown); self.reset()

    def reset(self):
        self.v = None; self.t_last = None; self.delta_hat = 0.0; self.S = 0.0; self.active = False; self.gate.reset(); self.gs = None

    def on_message(self, m):
        p = m["p"]
        if m["stream"] == "BVS1":
            self.v = p["speed"]; return []
        if self.v is None: return []
        t = p["timestamp"] / 1000.0
        if self.t_last is not None and t <= self.t_last: return []
        if self.gs is None: self.gs = p["wind_gust"]
        else: self.gs += (1.0 - math.exp(-(t - self.t_last) / self.tau_g)) * (p["wind_gust"] - self.gs)
        r = p["power_consumption"] / (self.th[0] + self.th[1] * self.v * self.v + self.th[2] * self.gs) - 1.0
        if self.t_last is None: self.delta_hat = 0.0
        else:
            a = 1.0 - math.exp(-(t - self.t_last) / self.tau); self.delta_hat += a * (r - self.delta_hat)
        self.t_last = t
        if t < self.warmup:
            return []
        if self.mode == "ewma":
            return [_alert("load_increase", delta=self.delta_hat)] if self.gate.update(t, self.delta_hat) == "on" else []
        self.S = max(0.0, self.S + r - self.k)
        if not self.active and self.S > self.h:
            self.active = True; return [_alert("load_increase", delta=self.delta_hat)]
        if self.active and self.delta_hat < self.k / 2:
            self.active = False; self.S = 0.0
        return []


HAIL_FEATS = ["att5", "att15", "imu5", "imu15", "pprob", "turb", "gust_f", "stab5"]


class HailFeatures:
    """Потоковые признаки града: скачки pitch/roll (робастный σ по окну 80 отсчётов), события IMU-ошибки 0x2101, прогноз НСУ-3."""

    def __init__(self, k=4.0, amin=2.5):
        self.k, self.amin = k, amin; self.reset()

    def reset(self):
        self.ap = deque(maxlen=80); self.ar = deque(maxlen=80); self.att = deque(); self.imu = deque(); self.stab = deque(); self.nsu = (0.2, 0.3); self.clock = 0.0

    def _spike(self, buf, x):
        ax = abs(x); buf.append(ax)
        if len(buf) < 20: return False
        sig = 1.4826 * float(np.median(buf)) / 0.6745 * 0.6745     # робастная оценка σ по медиане |x| (≈ медиана/0.6745)
        sig = float(np.median(buf)) / 0.6745
        return ax > self.k * sig and ax > self.amin

    def feed(self, m):
        s = m["stream"]; p = m["p"]
        if s == "BVS1":
            t = m["t_gen"]; self.clock = max(self.clock, t)
            if self._spike(self.ap, p["pitch"]) | self._spike(self.ar, p["roll"]): self.att.append(t)
        elif s == "BVS7":
            if 0x2101 in p["error_codes"]: self.imu.append(m["t_gen"])
        elif s == "NSU3":
            self.nsu = (p["precipitation_prob"], p["turbulence_index"])
        elif s == "BVS5":
            t = m["t_gen"]; self.clock = max(self.clock, t)
            self.stab.append((t, 1 if p["stabilization_status"] != "ok" else 0))
            for q, w in ((self.att, 15.0), (self.imu, 15.0)):
                while q and q[0] < t - w: q.popleft()
            while self.stab and self.stab[0][0] < t - 5.0: self.stab.popleft()
            a5 = sum(1 for x in self.att if x >= t - 5.0); i5 = sum(1 for x in self.imu if x >= t - 5.0)
            return np.array([a5, len(self.att), i5, len(self.imu), self.nsu[0], self.nsu[1], p["wind_gust"] - p["wind_speed"], sum(v for _, v in self.stab)], float)
        return None


class HailDetector(Detector):
    """Град: признаки HailFeatures → логрегрессия (numpy) → Gate. cols - используемые признаки (для абляций)."""
    name = "hail_fused"; streams = ("BVS1", "BVS5", "BVS7", "NSU3")

    def __init__(self, w, b, mu, sd, cols=None, on=0.6, off=0.3, n_on=2, n_off=5, cooldown=30.0):
        self.w, self.b, self.mu, self.sd = np.asarray(w), float(b), np.asarray(mu), np.asarray(sd)
        self.cols = list(range(len(HAIL_FEATS))) if cols is None else cols
        self.gate = Gate(on, off, n_on, n_off, cooldown); self.fe = HailFeatures(); self.reset()

    def reset(self):
        self.fe.reset(); self.gate.reset()

    def on_message(self, m):
        x = self.fe.feed(m)
        if x is None: return []
        z = (x[self.cols] - self.mu) / self.sd; pr = 1.0 / (1.0 + math.exp(-(float(self.w @ z) + self.b)))
        ev = self.gate.update(m["t_gen"], pr)
        return [_alert("hail", p=pr)] if ev == "on" else []


class DustThermalDetector(Detector):
    """Перегрев от пыли: остаток cpu_temp − (c0 + c1 (T−20)) (БВС-7, T из БВС-5): EWMA + CUSUM; плюс статус ESC=warn / код 0x3105."""
    name = "dust_thermal"; streams = ("BVS5", "BVS7")

    def __init__(self, model, tau=21.0, k=1.0, h=8.0):
        self.c0, self.c1, self.sig = model; self.tau, self.k, self.h = tau, k, h; self.reset()

    def reset(self):
        self.T = None; self.t_last = None; self.res = 0.0; self.S = 0.0; self.active = False

    def on_message(self, m):
        p = m["p"]
        if m["stream"] == "BVS5":
            self.T = p["temperature"]; return []
        if self.T is None: return []
        if 0x3105 in p["error_codes"] or p["component_status"].get("ESC") == "warn":
            if not self.active: self.active = True; return [_alert("dust_thermal", why="esc")]
            return []
        if not p["system_health"]: return []
        t = m["t_gen"]; r = p["system_health"]["cpu_temp"] - (self.c0 + self.c1 * (self.T - 20.0))
        if self.t_last is None: self.res = r
        else:
            a = 1.0 - math.exp(-(t - self.t_last) / self.tau); self.res += a * (r - self.res)
        self.t_last = t; self.S = max(0.0, self.S + r - self.k)
        if not self.active and self.S > self.h:
            self.active = True; return [_alert("dust_thermal", res=self.res)]
        if self.active and self.res < self.k:
            self.active = False; self.S = 0.0
        return []
