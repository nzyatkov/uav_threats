"""Стенд: прогон детекторов по потокам, замер времени обработки, задержки обнаружения и ложных тревог."""
import math
import time
from collections import deque
import numpy as np
import pandas as pd
from .streams import make_scenario, generate_bus, DT, NET
from .physics import wet_bulb_stull


class Detector:
    name = "base"
    streams = ()
    targets = ()                       # какие метки истины считаются «правильными» для этого детектора

    def reset(self):
        pass

    def on_message(self, m):
        return []


def _alert(kind, **kw):
    return dict(kind=kind, **kw)


def _dewpoint(T, RH):
    a, b = 17.625, 243.04
    g = math.log(max(RH, 0.1) / 100.0) + a * T / (b + T)
    return b * g / (a - g)


class CondensationDetector(Detector):
    """Угроза 28: физический канал (БВС-5): Magnus + инерционное звено линзы + Φ((D+m)/σ) + гистерезис. O(1) на сообщение."""
    name = "condensation"; streams = ("BVS5",); targets = ("lens_fog",)

    def __init__(self, tau=120.0, offset=0.0, margin=0.5, sigma=0.5, on=0.6, off=0.3):
        self.tau, self.offset, self.margin, self.sigma, self.on, self.off = tau, offset, margin, sigma, on, off
        self.reset()

    def reset(self):
        self.lens = None; self.t_last = None; self.state = False

    def on_message(self, m):
        p = m["p"]; t = p["timestamp"] / 1000.0; T = p["temperature"]; RH = p["humidity"]
        if self.lens is None:
            self.lens = T + self.offset; self.t_last = t; return []
        if t <= self.t_last:
            return []                                   # устаревшее/переупорядоченное сообщение
        a = 1.0 - math.exp(-(t - self.t_last) / self.tau); self.t_last = t
        self.lens += a * (T + self.offset - self.lens)
        D = _dewpoint(T, RH) - self.lens
        prob = 0.5 * (1.0 + math.erf((D + self.margin) / (self.sigma * math.sqrt(2.0))))
        out = []
        if not self.state and prob >= self.on:
            self.state = True; out.append(_alert("lens_fog_warning", p=prob))
        elif self.state and prob <= self.off:
            self.state = False
        return out


class PhantomClutterDetector(Detector):
    """Угрозы 1, 7, 8, 14: рост частоты «малых близких слабых» детекций lidar (БВС-4). Окно 10 с; БВС-5 как «тики» времени."""
    name = "phantom_clutter"; streams = ("BVS4", "BVS5"); targets = ("rain", "snow", "dust", "smoke")

    def __init__(self, win=10.0, on=0.5, off=0.2):
        self.win, self.on, self.off = win, on, off; self.reset()

    def reset(self):
        self.q = deque(); self.state = False

    def on_message(self, m):
        t = m["t_gen"]
        if m["stream"] == "BVS4":
            p = m["p"]
            if p["sensor_type"] == "lidar" and p["distance"] < 15.0 and p["size_estimate"] < 0.5 and p["detection_confidence"] < 0.6:
                self.q.append(t)
        while self.q and self.q[0] < t - self.win:
            self.q.popleft()
        rate = len(self.q) / self.win
        if not self.state and rate >= self.on:
            self.state = True; return [_alert("lidar_clutter", rate=rate)]
        if self.state and rate <= self.off:
            self.state = False
        return []


class PrecipRuleDetector(Detector):
    """Угрозы 7, 8: правило по RH и T_w (БВС-5): p = σ((RH−93)/2), тип по T_w; 3 подряд."""
    name = "precip_rule"; streams = ("BVS5",); targets = ("rain", "snow", "hail")

    def __init__(self, n_on=3, n_off=3):
        self.n_on, self.n_off = n_on, n_off; self.reset()

    def reset(self):
        self.k = 0; self.state = False

    def on_message(self, m):
        p = m["p"]; RH = p["humidity"]; T = p["temperature"]
        pr = 1.0 / (1.0 + math.exp(-(RH - 93.0) / 2.0))
        if not self.state:
            self.k = self.k + 1 if pr > 0.6 else 0
            if self.k >= self.n_on:
                self.state = True; self.k = 0
                tw = float(wet_bulb_stull(T, RH)) if RH < 99 else T - 0.3
                return [_alert("precip_snow" if tw < 1.0 else "precip_rain", p=pr)]
        else:
            self.k = self.k + 1 if pr < 0.3 else 0
            if self.k >= self.n_off:
                self.state = False; self.k = 0
        return []


class HailDetector(Detector):
    """Угроза 9: ≥3 события ошибки IMU (0x2101) в БВС-7 за 5 с либо ≥3 скачка |pitch|>4° в БВС-1 за 5 с."""
    name = "hail_events"; streams = ("BVS7", "BVS1"); targets = ("hail",)

    def __init__(self, win=5.0, need=3):
        self.win, self.need = win, need; self.reset()

    def reset(self):
        self.q = deque(); self.state = False

    def on_message(self, m):
        t = m["t_gen"]; p = m["p"]
        if m["stream"] == "BVS7" and 0x2101 in p["error_codes"]:
            self.q.append(t)
        elif m["stream"] == "BVS1" and abs(p["pitch"]) > 4.0:
            self.q.append(t)
        while self.q and self.q[0] < t - self.win:
            self.q.popleft()
        if not self.state and len(self.q) >= self.need:
            self.state = True; return [_alert("hail_events", n=len(self.q))]
        if self.state and len(self.q) == 0:
            self.state = False
        return []


def all_detectors():
    return [CondensationDetector(), PhantomClutterDetector(), PrecipRuleDetector(), HailDetector()]


def run_detectors(dets, msgs, rng=None):
    """Прогон по сообщениям в порядке прибытия. Возвращает alerts {имя: [...]}, times_ns {имя: [...]}."""
    rng = rng or np.random.default_rng(0)
    med, p99, _ = NET["BVS3"]
    for d in dets:
        d.reset()
    alerts = {d.name: [] for d in dets}; times = {d.name: [] for d in dets}
    by_stream = {}
    for d in dets:
        for s in d.streams:
            by_stream.setdefault(s, []).append(d)
    pc = time.perf_counter_ns
    for m in msgs:
        for d in by_stream.get(m["stream"], ()):
            t0 = pc(); out = d.on_message(m); dt = pc() - t0
            times[d.name].append(dt)
            for a in out:
                down = float(np.exp(rng.normal(np.log(med), np.log(p99 / med) / 2.326)))      # канал БВС-3 до НСУ
                a.update(t_gen=m["t_gen"], t_arrive=m["t_arrive"], proc_s=dt * 1e-9, t_deliver=m["t_arrive"] + dt * 1e-9 + down, down_s=down)
                alerts[d.name].append(a)
    return alerts, times


def evaluate(kinds, seeds, dets_factory=all_detectors, pre=60.0, post=30.0):
    """Таблица по (детектор, сценарий, зерно): detected, delay (от onset), false_alarm (в сценариях без цели)."""
    rows = []; lat = []
    for kind in kinds:
        for sd in seeds:
            cfg = make_scenario(kind, sd); msgs, tr, S = generate_bus(cfg)
            dets = dets_factory(); alerts, times = run_detectors(dets, msgs, np.random.default_rng(sd))
            for d in dets:
                tgt = np.zeros(len(S["t"]), bool)
                for k in d.targets:
                    tgt |= tr.get(k, np.zeros(len(S["t"]), bool))
                al = alerts[d.name]
                if tgt.any():
                    on, off = S["t"][np.argmax(tgt)], S["t"][len(tgt) - 1 - np.argmax(tgt[::-1])]
                    hit = [a for a in al if on - pre <= a["t_gen"] <= off + post]
                    rows.append(dict(детектор=d.name, сценарий=kind, seed=sd, цель=True, detected=bool(hit),
                                     delay=(min(a["t_deliver"] for a in hit) - on) if hit else np.nan, n_alerts=len(al)))
                else:
                    rows.append(dict(детектор=d.name, сценарий=kind, seed=sd, цель=False, detected=False, delay=np.nan, n_alerts=len(al),
                                     fa_per_hour=len(al) / (cfg.duration / 3600.0)))
                lat += [dict(детектор=d.name, proc_us=x / 1e3) for x in times[d.name]]
    return pd.DataFrame(rows), pd.DataFrame(lat)
