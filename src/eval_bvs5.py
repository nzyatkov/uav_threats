"""Оценка онлайн-детекторов на БВС-5/НСУ-3 по кэшу сценариев (чтобы перебор параметров был быстрым)."""
import numpy as np
import pandas as pd
from . import streams as st
from .streams import make_scenario, generate_bus, DT

KINDS = ["clear", "humid", "rain", "snow", "hail", "fog_cloud", "dust", "smoke", "lens_fog", "laser", "night_city"]


def label_at(S, i):
    I = S["I"]
    if I["hail"][i] > 0.2: return "hail"
    if I["snow"][i] > 0.2: return "snow"
    if I["rain"][i] > 0.2: return "rain"
    if I["fog_cloud"][i] > 0.2: return "fog_cloud"
    if I["dust"][i] > 0.2 or I["smoke"][i] > 0.2: return "aerosol"
    return "dry"


def segments(mask, gap=30.0):
    """Интервалы True на сетке DT (с объединением разрывов короче gap секунд). Возвращает [(t0, t1)]."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segs = []; s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if (i - prev) * DT > gap:
            segs.append((s * DT, prev * DT)); s = i
        prev = i
    segs.append((s * DT, prev * DT))
    return segs


def build_cache(kinds, seeds, loss5=None, bias=(0.0, 0.0), shift=None, streams=("BVS5", "NSU3")):
    """Кэш: [dict(kind, seed, msgs (только нужные потоки), truth, S-метки, duration)].
    loss5 - потеря БВС-5; bias=(dT, dRH) - смещение датчика температуры/влажности; shift - функция(cfg) для смены условий."""
    saved = st.NET["BVS5"]
    if loss5 is not None:
        st.NET["BVS5"] = (saved[0], saved[1], loss5)
    out = []
    try:
        for k in kinds:
            for sd in seeds:
                cfg = make_scenario(k, sd)
                if shift: shift(cfg)
                msgs, tr, S = generate_bus(cfg, skip=("BVS4", "BVS1", "BVS2", "BVS7", "NSU2"))
                msgs = [m for m in msgs if m["stream"] in streams]
                if bias != (0.0, 0.0):
                    for m in msgs:
                        if m["stream"] == "BVS5":
                            m["p"] = dict(m["p"]); m["p"]["temperature"] += bias[0]; m["p"]["humidity"] = float(np.clip(m["p"]["humidity"] + bias[1], 1, 100))
                out.append(dict(kind=k, seed=sd, msgs=msgs, truth=tr, S=S, duration=cfg.duration))
    finally:
        st.NET["BVS5"] = saved
    return out


def eval_condensation(factory, cache, pre=120.0, post=30.0):
    """Событийные метрики по метке lens_fog: recall, lead (с, >0 - тревога до onset), alerts/event, FA/час."""
    n_ev = hit = 0; leads = []; per_event = []; fa = 0; hours = 0.0; proc = 0
    for c in cache:
        d = factory(); d.reset()
        alerts = []
        for m in c["msgs"]:
            if m["stream"] in d.streams:
                for a in d.on_message(m):
                    alerts.append(m["t_gen"])
        segs = segments(c["truth"]["lens_fog"]); hours += c["duration"] / 3600.0
        inwin = np.zeros(len(alerts), bool)
        for (t0, t1) in segs:
            n_ev += 1
            a_in = [t for t in alerts if t0 - pre <= t <= t1 + post]
            for j, t in enumerate(alerts):
                if t0 - pre <= t <= t1 + post: inwin[j] = True
            if a_in:
                hit += 1; leads.append(t0 - min(a_in)); per_event.append(len(a_in))
        fa += int((~inwin).sum())
    return dict(recall=hit / max(n_ev, 1), lead_med=float(np.median(leads)) if leads else np.nan, alerts_per_event=float(np.mean(per_event)) if per_event else np.nan,
                fa_per_hour=fa / hours, n_events=n_ev)


def build_rows(cache, use_nsu=True):
    """Обучающие строки: признаки по потоковому извлечению + метка по скрытому состоянию на t_gen."""
    from .detectors_bvs5 import Bvs5Features
    X, y, meta = [], [], []
    for ci, c in enumerate(cache):
        fe = Bvs5Features(use_nsu)
        for m in c["msgs"]:
            if m["stream"] == "NSU3":
                if use_nsu: fe.update_nsu(m)
            else:
                x = fe.update_bvs5(m)
                if x is not None:
                    X.append(x); y.append(label_at(c["S"], int(m["t_gen"] / DT))); meta.append((ci, m["t_gen"]))
    return np.array(X), np.array(y), meta


def dwell_states(classes, times, dwell=3):
    """Подтверждение смены класса (как в ConditionDetector): переключаемся после dwell одинаковых предсказаний подряд.
    Возвращает список переходов [(t, новый_класс)]."""
    state, cand, k, out = "dry", None, 0, []
    for c, t in zip(classes, times):
        if c == state:
            cand, k = None, 0; continue
        if c == cand: k += 1
        else: cand, k = c, 1
        if k >= dwell:
            state, cand, k = c, None, 0; out.append((t, c))
    return out


def events_from_alerts(cache, trans_per_scenario, pre=30.0, post=30.0, tgt=None):
    """Событийные метрики по переходам состояния: recall и задержка по классам, ложные в час, переключений в час."""
    tgt = tgt or {"rain": ["rain"], "snow": ["snow"], "hail": ["hail"], "fog_cloud": ["fog_cloud"], "aerosol": ["dust", "smoke"]}
    rows = {k: dict(n=0, hit=0, delays=[], fa=0) for k in tgt}; hours = 0.0; switches = 0
    for c, trans in zip(cache, trans_per_scenario):
        hours += c["duration"] / 3600.0; switches += len(trans)
        alerts = [(t, k) for t, k in trans if k != "dry"]; used = [False] * len(alerts)
        for cls, kinds in tgt.items():
            mask = np.zeros(len(c["S"]["t"]), bool)
            for k in kinds: mask |= c["truth"][k]
            for (t0, t1) in segments(mask):
                rows[cls]["n"] += 1
                a_in = [j for j, (t, k) in enumerate(alerts) if k == cls and t0 - pre <= t <= t1 + post]
                if a_in:
                    rows[cls]["hit"] += 1; rows[cls]["delays"].append(alerts[a_in[0]][0] - t0)
                for j in a_in: used[j] = True
        for j, (t, k) in enumerate(alerts):
            if not used[j]: rows[k]["fa"] += 1
    out = pd.DataFrame([dict(класс=k, событий=v["n"], recall=v["hit"] / max(v["n"], 1), задержка_медиана_с=float(np.median(v["delays"])) if v["delays"] else np.nan,
                             ложных_в_час=v["fa"] / hours) for k, v in rows.items()]).set_index("класс")
    return out, switches / hours


def eval_condition_events(factory, cache, **kw):
    """Прогон онлайн-детектора по кэшу (медленно для тяжёлых моделей)."""
    trans = []
    for c in cache:
        d = factory(); d.reset(); tr = []; prev = d.state
        for m in c["msgs"]:
            if m["stream"] in d.streams:
                d.on_message(m)
                if d.state != prev: tr.append((m["t_gen"], d.state)); prev = d.state
        trans.append(tr)
    return events_from_alerts(cache, trans, **kw)
