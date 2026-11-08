"""Кэш сценариев (БВС-1, БВС-5, БВС-7, НСУ-3 в «лёгком» виде) и событийная оценка детекторов мощности/устойчивости."""
import numpy as np
import pandas as pd
from .streams import make_scenario, generate_bus, DT
from .eval_bvs5 import segments

KINDS3 = ["clear", "humid", "rain", "snow", "hail", "fog_cloud", "dust", "smoke", "lens_fog", "laser", "night_city"]
KEEP = {"BVS1": ("timestamp", "pitch", "roll", "speed"),
        "BVS5": ("timestamp", "temperature", "humidity", "wind_speed", "wind_gust", "power_consumption", "stabilization_status"),
        "BVS7": ("system_health", "component_status", "error_codes"),
        "NSU3": ("precipitation_prob", "turbulence_index", "visibility_m")}


def build_cache3(kinds, seeds, modify=None, net=None):
    """modify(cfg) - изменение сценария (стресс-тесты); net - {поток: (медиана, p99, потеря)} временная подмена сетевой модели."""
    from . import streams as st
    saved = {k: st.NET[k] for k in (net or {})}
    st.NET.update(net or {})
    out = []
    try:
        for k in kinds:
            for sd in seeds:
                cfg = make_scenario(k, sd)
                if modify: modify(cfg)
                msgs, tr, S = generate_bus(cfg, skip=("BVS4", "BVS2", "NSU2"))
                slim = [dict(stream=m["stream"], t_gen=m["t_gen"], t_arrive=m["t_arrive"], p={f: m["p"][f] for f in KEEP[m["stream"]]})
                        for m in msgs if m["stream"] in KEEP]
                out.append(dict(kind=k, seed=sd, msgs=slim, truth=tr, S=S, duration=cfg.duration))
    finally:
        st.NET.update(saved)
    return out


def load_mask(c, thr=0.025):
    return c["S"]["delta"] > thr


def eval_events(factory, cache, target, kind_of_alert, pre=0.0, post=60.0):
    """Событийные метрики: target(c) -> маска истинных интервалов; kind_of_alert - какие алерты считать. Возвращает dict."""
    n = hit = fa = 0; delays = []; per = []; hours = 0.0
    for c in cache:
        d = factory(); d.reset(); al = []
        for m in c["msgs"]:
            if m["stream"] in d.streams:
                for a in d.on_message(m):
                    if a["kind"] == kind_of_alert: al.append(m["t_gen"])
        hours += c["duration"] / 3600.0; used = np.zeros(len(al), bool)
        for (t0, t1) in segments(target(c)):
            n += 1; a_in = [j for j, t in enumerate(al) if t0 - pre <= t <= t1 + post]
            if a_in: hit += 1; delays.append(al[a_in[0]] - t0); per.append(len(a_in))
            for j in a_in: used[j] = True
        fa += int((~used).sum())
    return dict(событий=n, recall=hit / max(n, 1), задержка_медиана_с=float(np.median(delays)) if delays else np.nan,
                алертов_на_событие=float(np.mean(per)) if per else np.nan, ложных_в_час=fa / hours)


def build_hail_rows(cache):
    from .detectors_power import HailFeatures
    X, y, meta = [], [], []
    for ci, c in enumerate(cache):
        fe = HailFeatures()
        for m in c["msgs"]:
            x = fe.feed(m)
            if x is not None:
                X.append(x); y.append(bool(c["S"]["I"]["hail"][int(m["t_gen"] / DT)] > 0.2)); meta.append((ci, m["t_gen"]))
    return np.array(X), np.array(y), meta
