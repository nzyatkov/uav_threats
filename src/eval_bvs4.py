"""Оценка детекторов на БВС-4: кэш сценариев, метки условий, истина слепоты сенсоров, событийные метрики."""
import numpy as np
import pandas as pd
from .streams import make_scenario, generate_bus, DT
from .eval_bvs5 import segments
from .detectors_bvs4 import Bvs4Features, BVS4_FEATS

KINDS4 = ["clear", "humid", "rain", "snow", "hail", "fog_cloud", "dust", "smoke", "lens_fog", "laser", "night_city"]
CLASSES9 = ["dry", "rain", "snow", "fog_cloud", "aerosol", "hail", "laser", "night_city"]
TGT9 = {"rain": ["rain"], "snow": ["snow"], "hail": ["hail"], "fog_cloud": ["fog_cloud"], "aerosol": ["dust", "smoke"], "laser": ["laser"], "night_city": ["night_city"]}


def epoch_of(cfg):
    return "2026-11-07 22:00" if cfg.night else "2026-11-07 12:00"


def label9(S, i):
    I = S["I"]
    if I["laser"][i] > 0.5: return "laser"
    if I["hail"][i] > 0.2: return "hail"
    if I["snow"][i] > 0.2: return "snow"
    if I["rain"][i] > 0.2: return "rain"
    if I["fog_cloud"][i] > 0.2: return "fog_cloud"
    if I["dust"][i] > 0.2 or I["smoke"][i] > 0.2: return "aerosol"
    if S["night_city"][i] > 0.3: return "night_city"
    return "dry"


def blind_truth(S):
    """Истина слепоты (по скрытым параметрам генератора): lidar - дальность < 100 м; camera - дальность < 80 м (туман, запотевание, лазер)."""
    V, lf, las = S["V"], S["lens_f"], S["laser_on"]
    r_l = np.minimum(150.0, 0.15 * V)
    r_c = np.minimum(120.0, 0.4 * V) * (1 - 0.9 * lf) * np.where(las, 0.15, 1.0)
    return {"lidar": r_l < 100.0, "camera": r_c < 80.0}


def build_cache4(kinds, seeds, obstacle_rate=None, modify=None):
    out = []
    for k in kinds:
        for sd in seeds:
            cfg = make_scenario(k, sd)
            if obstacle_rate: cfg.obstacle_rate = obstacle_rate
            if modify: modify(cfg)
            msgs, tr, S = generate_bus(cfg, skip=("BVS2", "BVS7"))
            k1 = 0; keep = []
            for m in msgs:                                    # БВС-1 прореживаем до ~1 Гц: из него нужна только позиция (память)
                if m["stream"] == "BVS1":
                    k1 += 1
                    if k1 % 8: continue
                keep.append(m)
            msgs = keep
            out.append(dict(kind=k, seed=sd, msgs=msgs, truth=tr, S=S, duration=cfg.duration, epoch=epoch_of(cfg), blind=blind_truth(S)))
    return out


def build_rows4(cache, use_nsu=True):
    X, y, meta = [], [], []
    for ci, c in enumerate(cache):
        fe = Bvs4Features(c["epoch"], use_nsu=use_nsu)
        for m in c["msgs"]:
            x = fe.feed(m)
            if x is not None:
                X.append(x); y.append(label9(c["S"], int(m["t_gen"] / DT))); meta.append((ci, m["t_gen"]))
    return np.array(X), np.array(y), meta


def eval_blind(factory, cache, post=60.0):
    """Событийные метрики слепоты: для каждого сенсора - recall, медианная задержка (с), ложные включения в час."""
    out = {}
    for sen in ("lidar", "camera"):
        n = hit = fa = 0; delays = []; hours = 0.0
        for c in cache:
            d = factory(c["epoch"]); d.reset(); al = []
            for m in c["msgs"]:
                if m["stream"] in d.streams:
                    for a in d.on_message(m):
                        if a["kind"] == f"blind_{sen}": al.append(m["t_gen"])
            hours += c["duration"] / 3600.0; used = np.zeros(len(al), bool)
            for (t0, t1) in segments(c["blind"][sen]):
                n += 1; a_in = [j for j, t in enumerate(al) if t0 <= t <= t1 + post]
                if a_in: hit += 1; delays.append(al[a_in[0]] - t0)
                for j in a_in: used[j] = True
            fa += int((~used).sum())
        out[sen] = dict(событий=n, recall=hit / max(n, 1), задержка_медиана_с=float(np.median(delays)) if delays else np.nan, ложных_в_час=fa / hours)
    return pd.DataFrame(out).T
