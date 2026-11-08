"""Онлайн-признаки и детекторы по БВС-4 (lidar / radar / camera): перекрёстное подтверждение, слепота сенсоров, контекст ночи и зон.

Идея: настоящее препятствие видят несколько сенсоров согласованно (по времени, пеленгу, дальности), «фантом» помехи - только один.
Поэтому доля НЕподтверждённых детекций сенсора - признак помехи именно этого сенсора (lidar: осадки/пыль/дым, camera: ночные огни/лазер).
Слепоту сенсора выявляем по дальности первого обнаружения относительно radar на треках препятствий (событийно)."""
import math
from collections import deque
import numpy as np
import pandas as pd
from .harness import Detector, _alert
from .alerts import Gate
from .detectors_bvs5 import Bvs5Features, FEATS
from .light_features import solar_elevation
from .streams import LAT0, LON0

SENS = ("lidar", "radar", "camera")
F4 = [f"{k}_{s}" for s in SENS for k in ("n", "u", "small", "near", "conf", "deg", "fail", "gap")]
EXTRA = ["night", "zone", "r_lidar", "r_cam", "age_ratio"]
BVS4_FEATS = FEATS + F4 + EXTRA


class Bvs4Features:
    """Потоковые признаки: БВС-5/НСУ-3 (18) + окно 10 с по БВС-4 (24) + контекст (5). Выдаёт вектор на каждом сообщении БВС-5 (1 Гц)."""

    def __init__(self, epoch="2026-11-07 12:00", win=10.0, hold=0.5, use_nsu=True, keep_log=False):
        self.epoch = pd.Timestamp(epoch); self.win, self.hold, self.use_nsu, self.keep_log = win, hold, use_nsu, keep_log; self.reset()

    def reset(self):
        self.fe5 = Bvs5Features(self.use_nsu); self.clock = 0.0; self.log = []
        self.recent = deque(); self.pending = deque(); self.fin = deque()
        self.last_seen = {s: -1e9 for s in SENS}; self.zones = []; self.pos = (LAT0, LON0)
        self.track = None; self.buf = {"lidar": deque(), "camera": deque()}
        self.ratio = {"lidar": 1.0, "camera": 1.0}; self.t_ratio = -1e9; self.night = (-1e9, False); self.last_first = {"lidar": None, "camera": None}

    # --- события потоков ---
    def _gate(self, a, b):
        return abs(a["t"] - b["t"]) <= 0.25 and abs(a["b"] - b["b"]) <= 4.0 and abs(a["d"] - b["d"]) <= 4.0

    def _on_b4(self, m):
        p = m["p"]; s = p["sensor_type"]; t = m["t_gen"]; self.clock = max(self.clock, t); self.last_seen[s] = max(self.last_seen[s], t)
        det = dict(t=t, s=s, d=p["distance"], b=p["bearing"], size=p["size_estimate"], conf=p["detection_confidence"], h=p["sensor_health_status"], corr=False, ph=p.get("_phantom"))   # ph - метка истины, ТОЛЬКО для оценки
        for r in self.recent:
            if r["s"] != s and self._gate(r, det):
                r["corr"] = True; det["corr"] = True
        self.recent.append(det); self.pending.append(det)
        while self.recent and self.recent[0]["t"] < self.clock - 1.5:
            self.recent.popleft()
        if s == "radar":
            self._radar_track(det)
        else:
            self.buf[s].append(det)
            while self.buf[s] and self.buf[s][0]["t"] < self.clock - 25.0:
                self.buf[s].popleft()

    def _radar_track(self, det):
        tr = self.track
        if tr is not None and det["t"] - tr["last"] < 1.5 and abs(det["b"] - tr["b"]) < 4.0:
            tr["dets"].append(det); tr["last"] = max(tr["last"], det["t"])
        else:
            self._finalize_track(); self.track = dict(dets=[det], last=det["t"], b=det["b"])

    def _finalize_track(self):
        tr = self.track; self.track = None
        if tr is None or len(tr["dets"]) < 8:
            return
        ds = sorted(tr["dets"], key=lambda x: x["t"]); d0 = ds[0]["d"]; t0, t1 = ds[0]["t"], ds[-1]["t"]
        if d0 < 100.0:
            return
        for s in ("lidar", "camera"):
            cand = [x for x in self.buf[s] if t0 - 0.3 <= x["t"] <= t1 + 0.3 and abs(x["b"] - tr["b"]) < 4.0 and x["d"] > 15.0 and x["corr"]]
            r = min(cand, key=lambda x: x["t"])["d"] / d0 if cand else 0.0
            self.last_first[s] = (min(cand, key=lambda x: x["t"])["d"] if cand else 0.0, d0)
            self.ratio[s] = 0.5 * self.ratio[s] + 0.5 * min(r, 1.0)
        self.t_ratio = self.clock

    def _night(self):
        if self.clock - self.night[0] > 30.0:
            self.night = (self.clock, solar_elevation(self.epoch + pd.Timedelta(seconds=self.clock), LAT0, LON0) < -6.0)
        return self.night[1]

    def _zone_active(self):
        for z in self.zones:
            dx = (self.pos[1] - z["lon"]) * math.cos(math.radians(self.pos[0])) * 111320.0; dy = (self.pos[0] - z["lat"]) * 110540.0
            if math.hypot(dx, dy) < z["r"] and z["t0"] <= self.clock * 1000 <= z["t1"]:
                return 1.0
        return 0.0

    def feed(self, m):
        s = m["stream"]
        if s == "BVS4": self._on_b4(m)
        elif s == "BVS1": self.pos = (m["p"]["lat"], m["p"]["lon"]); self.clock = max(self.clock, m["t_gen"])
        elif s == "NSU2":
            g = m["p"]["geometry"]; w = m["p"]["validity_window"]
            self.zones.append(dict(lat=g["lat"], lon=g["lon"], r=g["radius_m"], t0=w["start_ts"], t1=w["end_ts"]))
        elif s == "NSU3":
            if self.use_nsu: self.fe5.update_nsu(m)
        elif s == "BVS5":
            self.clock = max(self.clock, m["t_gen"])
            x5 = self.fe5.update_bvs5(m)
            if x5 is None: return None
            while self.pending and self.pending[0]["t"] < self.clock - self.hold:
                d = self.pending.popleft(); self.fin.append(d)
                if self.keep_log: self.log.append(d)
            while self.fin and self.fin[0]["t"] < self.clock - self.win:
                self.fin.popleft()
            if self.track is not None and self.clock - self.track["last"] > 2.5:
                self._finalize_track()
            f4 = []
            for sen in SENS:
                its = [d for d in self.fin if d["s"] == sen]; n = len(its)
                if n:
                    f4 += [n, sum(1 for d in its if not d["corr"]), sum(1 for d in its if d["size"] < 0.5) / n, sum(1 for d in its if d["d"] < 15.0) / n,
                           sum(d["conf"] for d in its) / n, sum(1 for d in its if d["h"] == "DEGRADED") / n, sum(1 for d in its if d["h"] == "FAIL"),
                           min(self.clock - self.last_seen[sen], 60.0)]
                else:
                    f4 += [0, 0, 0.0, 0.0, 0.0, 0.0, 0, min(self.clock - self.last_seen[sen], 60.0)]
            ex = [float(self._night()), self._zone_active(), self.ratio["lidar"], self.ratio["camera"], min(self.clock - self.t_ratio, 300.0)]
            return np.concatenate([x5, np.array(f4, float), np.array(ex, float)])
        return None


class Condition4Detector(Detector):
    """Классификатор условий по признакам БВС-5 + НСУ-3 + БВС-4 + контекст: модель + подтверждение смены класса (dwell)."""
    name = "condition4"; streams = ("BVS1", "BVS4", "BVS5", "NSU2", "NSU3")

    def __init__(self, model, epoch="2026-11-07 12:00", dwell=3, cols=None):
        self.model, self.dwell, self.cols = model, dwell, cols; self.fe = Bvs4Features(epoch); self.reset()

    def reset(self):
        self.fe.reset(); self.state = "dry"; self.cand = None; self.k = 0

    def on_message(self, m):
        x = self.fe.feed(m)
        if x is None: return []
        if self.cols is not None: x = x[self.cols]
        pr = self.model.predict_proba(x); c = self.model.classes[int(np.argmax(pr))]
        if c == self.state: self.cand = None; self.k = 0; return []
        if c == self.cand: self.k += 1
        else: self.cand, self.k = c, 1
        if self.k >= self.dwell:
            self.state = c; self.cand = None; self.k = 0
            if c != "dry": return [_alert("cond_" + c, p=float(pr.max()))]
        return []


class BlindnessMonitor(Detector):
    """Слепота lidar/camera: EWMA отношения дальности первого обнаружения к radar на треках препятствий (событийно).
    Пороги раздельные: у камеры нормальная дальность короче (отношение ≈ 0,8), у lidar ≈ 1,0."""
    name = "blindness"; streams = ("BVS1", "BVS4", "BVS5", "NSU2", "NSU3")

    def __init__(self, thr=0.7, thr_cam=None, on_n=1, off_n=1, epoch="2026-11-07 12:00"):
        self.thr = {"lidar": thr, "camera": thr if thr_cam is None else thr_cam}
        self.fe = Bvs4Features(epoch, use_nsu=False)
        self.g = {s: Gate(1 - self.thr[s], 1 - self.thr[s] - 0.15, on_n, off_n) for s in ("lidar", "camera")}; self.reset()

    def reset(self):
        self.fe.reset(); [g.reset() for g in self.g.values()]; self.flags = {"lidar": False, "camera": False}

    def on_message(self, m):
        x = self.fe.feed(m)
        if x is None: return []
        out = []; t = m["t_gen"]
        for s in ("lidar", "camera"):
            ev = self.g[s].update(t, 1.0 - self.fe.ratio[s])
            if ev == "on": self.flags[s] = True; out.append(_alert(f"blind_{s}", ratio=self.fe.ratio[s]))
            elif ev == "off": self.flags[s] = False
        return out
