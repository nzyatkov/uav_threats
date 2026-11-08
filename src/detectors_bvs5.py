"""Онлайн-детекторы на БВС-5 / НСУ-3: конденсат (физика линзы) и классификатор метеоусловий. O(1) на сообщение."""
import math
from collections import deque
import numpy as np
from .harness import Detector, _alert, _dewpoint
from .alerts import Gate

CLASSES = ["dry", "rain", "snow", "fog_cloud", "aerosol", "hail"]


def tw_scalar(T, RH):
    """Температура мокрого термометра (Stull, 2011), скалярная версия."""
    RH = min(max(RH, 5.0), 99.0)
    return (T * math.atan(0.151977 * math.sqrt(RH + 8.313659)) + math.atan(T + RH) - math.atan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * math.atan(0.023101 * RH) - 4.686035)


class CondensationV2(Detector):
    """Физический канал конденсата с подавлением повторов: EMA оценки D = T_d − T_lens, Φ((D+m)/σ), Gate(on/off, n, cooldown)."""
    name = "condensation_v2"; streams = ("BVS5",); targets = ("lens_fog",)

    def __init__(self, tau=120.0, offset=-1.0, margin=0.5, sigma=0.5, alpha=1.0, on=0.6, off=0.3, n_on=1, n_off=1, cooldown=0.0):
        self.tau, self.offset, self.margin, self.sigma, self.alpha = tau, offset, margin, sigma, alpha
        self.gate = Gate(on, off, n_on, n_off, cooldown); self.reset()

    def reset(self):
        self.lens = None; self.t_last = None; self.Ds = None; self.gate.reset()

    def on_message(self, m):
        p = m["p"]; t = p["timestamp"] / 1000.0; T = p["temperature"]; RH = p["humidity"]
        if self.lens is None:
            self.lens = T + self.offset; self.t_last = t; return []
        if t <= self.t_last:
            return []
        a = 1.0 - math.exp(-(t - self.t_last) / self.tau); self.t_last = t
        self.lens += a * (T + self.offset - self.lens)
        D = _dewpoint(T, RH) - self.lens
        self.Ds = D if self.Ds is None else self.Ds + self.alpha * (D - self.Ds)
        prob = 0.5 * (1.0 + math.erf((self.Ds + self.margin) / (self.sigma * math.sqrt(2.0))))
        self.last_prob = prob
        ev = self.gate.update(t, prob)
        return [_alert("lens_fog_warning", p=prob)] if ev == "on" else []


FEATS = ["T", "RH", "spread", "Tw", "wind", "gust", "gust_f", "T_ema", "RH_ema", "spread_ema", "dT_30", "dRH_30", "dspread_30", "P_ema",
         "logV", "pprob", "turb", "nsu_age"]
FEATS_BVS5 = FEATS[:14]


class Bvs5Features:
    """Потоковое вычисление признаков: окно 30 с по БВС-5 (по timestamp) + последние значения НСУ-3."""

    def __init__(self, use_nsu=True):
        self.use_nsu = use_nsu; self.reset()

    def reset(self):
        self.win = deque(); self.ema = None; self.t_last = None; self.nsu = None; self.nsu_t = None

    def update_nsu(self, m):
        p = m["p"]; self.nsu = (math.log(max(p["visibility_m"], 10.0)), p["precipitation_prob"], p["turbulence_index"]); self.nsu_t = m["t_gen"]

    def update_bvs5(self, m):
        p = m["p"]; t = p["timestamp"] / 1000.0
        if self.t_last is not None and t <= self.t_last:
            return None
        T, RH = p["temperature"], p["humidity"]; Td = _dewpoint(T, RH); sp = T - Td
        Tw = tw_scalar(T, RH) if RH < 99 else T - 0.3
        P = p["power_consumption"]
        if self.ema is None:
            self.ema = [T, RH, sp, P]
        else:
            a = 1 - math.exp(-(t - self.t_last) / 30.0)
            self.ema = [e + a * (x - e) for e, x in zip(self.ema, (T, RH, sp, P))]
        self.t_last = t
        self.win.append((t, T, RH, sp))
        while self.win and self.win[0][0] < t - 30.0:
            self.win.popleft()
        d = [0.0, 0.0, 0.0]
        if len(self.win) >= 5:
            ts = np.array([w[0] for w in self.win]); ts = ts - ts.mean(); den = float((ts * ts).sum()) + 1e-9
            for k in range(3):
                ys = np.array([w[1 + k] for w in self.win]); d[k] = float((ts * (ys - ys.mean())).sum() / den * 30.0)
        w, g = p["wind_speed"], p["wind_gust"]
        row = [T, RH, sp, Tw, w, g, g - w, self.ema[0], self.ema[1], self.ema[2], d[0], d[1], d[2], self.ema[3]]
        if self.use_nsu:
            if self.nsu is None:
                row += [math.log(30000.0), 0.2, 0.3, 999.0]
            else:
                row += [self.nsu[0], self.nsu[1], self.nsu[2], t - self.nsu_t]
        return np.array(row)


class NumpyLogReg:
    """Многоклассовая логрегрессия как чистый numpy (мкс на вызов)."""

    def __init__(self, sk_pipeline):
        sc, lr = sk_pipeline.named_steps["standardscaler"], sk_pipeline.named_steps["logisticregression"]
        self.mu, self.sd, self.W, self.b, self.classes = sc.mean_, sc.scale_, lr.coef_, lr.intercept_, list(lr.classes_)

    def predict_proba(self, x):
        z = self.W @ ((x - self.mu) / self.sd) + self.b; z = z - z.max(); e = np.exp(z); return e / e.sum()


class SkWrap:
    def __init__(self, model):
        self.m = model; self.classes = list(model.classes_)

    def predict_proba(self, x):
        return self.m.predict_proba(x.reshape(1, -1))[0]


class ConditionDetector(Detector):
    """Классификатор метеоусловий онлайн: признаки по потокам + модель + подтверждение смены класса (dwell). Алерт при входе в не-dry класс."""
    name = "condition"; streams = ("BVS5", "NSU3")

    def __init__(self, model, use_nsu=True, dwell=3):
        self.model, self.use_nsu, self.dwell = model, use_nsu, dwell; self.fe = Bvs5Features(use_nsu); self.reset()

    def reset(self):
        self.fe.reset(); self.state = "dry"; self.cand = None; self.k = 0; self.last_p = None

    def on_message(self, m):
        if m["stream"] == "NSU3":
            if self.use_nsu:
                self.fe.update_nsu(m)
            return []
        x = self.fe.update_bvs5(m)
        if x is None:
            return []
        pr = self.model.predict_proba(x); self.last_p = pr
        c = self.model.classes[int(np.argmax(pr))]
        if c == self.state:
            self.cand = None; self.k = 0; return []
        if c == self.cand:
            self.k += 1
        else:
            self.cand, self.k = c, 1
        if self.k >= self.dwell:
            self.state = c; self.cand = None; self.k = 0
            if c != "dry":
                return [_alert("cond_" + c, p=float(pr.max()))]
        return []
