"""METAR/ASOS: загрузка из Iowa Environmental Mesonet и разметка классов погоды.

Классы (приоритет сверху вниз): hail_ts > snow > rain > fog > aerosol > dry.
Если сети нет, load_or_fallback() вернёт СИНТЕТИЧЕСКИЙ ряд (только для проверки пайплайна!).
"""
import io
import os
import re
import numpy as np
import pandas as pd

from .physics import dewpoint, wet_bulb_stull

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CLASSES = ["dry", "rain", "snow", "fog", "aerosol", "hail_ts"]
_PRIORITY = {"dry": 0, "aerosol": 1, "fog": 2, "rain": 3, "snow": 4, "hail_ts": 5}
_RANK2CLASS = {v: k for k, v in _PRIORITY.items()}


def label_from_wxcodes(wx):
    """Класс по строке present weather METAR (например '-RA BR', 'TSRA', 'M')."""
    if not isinstance(wx, str) or wx.strip() in ("", "M"):
        return "dry"
    toks = [re.sub(r"^[-+]|^VC", "", t) for t in wx.split()]
    s = " ".join(toks)
    if "GR" in s or "GS" in s or "TS" in s:
        return "hail_ts"
    if any(c in s for c in ("SN", "SG", "PL", "IC")):
        return "snow"
    if any(c in s for c in ("RA", "DZ", "UP")):
        return "rain"
    if "FG" in s:
        return "fog"
    if any(c in s for c in ("HZ", "DU", "SA", "FU", "VA")):
        return "aerosol"
    return "dry"   # BR (дымка) и прочее считаем «сухо»


def download_iem(station, year, cache_dir="data"):
    """Скачать METAR+SPECI станции за год (кэшируется в CSV)."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"iem_{station}_{year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    params = {
        "station": station,
        "data": ["tmpc", "dwpc", "relh", "sknt", "gust", "vsby", "wxcodes"],
        "year1": year, "month1": 1, "day1": 1,
        "year2": year + 1, "month2": 1, "day2": 1,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no", "report_type": [3, 4],
    }
    r = requests.get(IEM_URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=["M"], keep_default_na=False,
                     dtype={"wxcodes": str}, comment="#")
    df.to_csv(path, index=False)
    return df


def hourly_table(raw):
    """Сырые наблюдения -> почасовая таблица с признаками и классом."""
    raw = raw.copy()
    raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
    for c in ["tmpc", "dwpc", "relh", "sknt", "gust", "vsby"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["rank"] = raw["wxcodes"].map(lambda s: _PRIORITY[label_from_wxcodes(s)])
    raw["hour"] = raw["valid"].dt.floor("h")
    g = raw.groupby("hour").agg(tmpc=("tmpc", "mean"), dwpc=("dwpc", "mean"),
                                relh=("relh", "mean"), sknt=("sknt", "mean"),
                                gust=("gust", "max"), vsby=("vsby", "min"),
                                rank=("rank", "max"))
    g = g.asfreq("h")
    g["label"] = g["rank"].map(_RANK2CLASS)
    g["wind_ms"] = g["sknt"] * 0.5144
    g["gust_ms"] = g["gust"].fillna(g["sknt"]) * 0.5144
    g["vis_m"] = g["vsby"] * 1609.34          # IEM отдаёт мили
    return g.drop(columns=["rank", "sknt", "gust", "vsby"])


def add_features(g):
    """Признаки, доступные из БВС-5 (+ vis_m из НСУ-3)."""
    g = g.copy()
    g["relh"] = g["relh"].fillna(
        100 * np.exp(17.625 * g["dwpc"] / (243.04 + g["dwpc"]) - 17.625 * g["tmpc"] / (243.04 + g["tmpc"])))
    g["Td"] = dewpoint(g["tmpc"], g["relh"])
    g["Tw"] = wet_bulb_stull(g["tmpc"], g["relh"])
    g["spread"] = g["tmpc"] - g["Td"]
    g["dT_1h"] = g["tmpc"].diff(1)
    g["dT_3h"] = g["tmpc"].diff(3)
    g["dspread_1h"] = g["spread"].diff(1)
    g["gust_factor"] = g["gust_ms"] - g["wind_ms"]
    return g


def load_stations(stations, years, cache_dir="data"):
    frames = []
    for st in stations:
        for y in years:
            raw = download_iem(st, y, cache_dir)
            h = add_features(hourly_table(raw))
            h["station"] = st
            frames.append(h)
    return pd.concat(frames)


# --- синтетический запасной вариант (метки порождены правилами!) ---------------

def synthetic_fallback(stations, years, seed=0):
    """Правдоподобный по виду, но КРУГОВОЙ ряд: метки получены из правил по T/RH.
    Годится только для проверки, что код работает без сети."""
    rng = np.random.default_rng(seed)
    frames = []
    for si, st in enumerate(stations):
        idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h", tz="UTC")
        n = len(idx)
        doy = idx.dayofyear.values
        hr = idx.hour.values
        T = 9 - 11 * np.cos(2 * np.pi * (doy - 15) / 365) + 3 * np.sin(2 * np.pi * (hr - 9) / 24)
        T = T + np.convolve(rng.normal(0, 1.2, n), np.ones(24) / 24, "same") * 6 + si
        spread = np.abs(2.5 + np.convolve(rng.normal(0, 1, n), np.ones(12) / 12, "same") * 8
                        + rng.normal(0, 0.8, n))
        Td = T - spread
        wind = np.abs(rng.gamma(2.0, 1.6, n))
        gust = wind + np.abs(rng.normal(0, 1.5, n))
        RH = 100 * np.exp(17.625 * Td / (243.04 + Td) - 17.625 * T / (243.04 + T))
        p_pr = 1 / (1 + np.exp(-(RH - 93) / 2.0))
        precip = rng.random(n) < p_pr * 0.7
        Tw = wet_bulb_stull(T, RH)
        label = np.full(n, "dry", dtype=object)
        label[precip & (Tw < 1.0)] = "snow"
        label[precip & (Tw >= 1.0)] = "rain"
        fog = (~precip) & (spread < 0.8) & (wind < 3)
        label[fog] = "fog"
        ts = precip & (T > 18) & (rng.random(n) < 0.05)
        label[ts] = "hail_ts"
        vis = np.where(label == "fog", rng.uniform(100, 900, n), rng.uniform(4000, 20000, n))
        vis = np.where(np.isin(label, ["rain", "snow"]), rng.uniform(1500, 8000, n), vis)
        g = pd.DataFrame(dict(tmpc=T, dwpc=Td, relh=RH, wind_ms=wind, gust_ms=gust,
                              vis_m=vis, label=label), index=idx)
        g.index.name = "hour"
        g = add_features(g)
        g["station"] = st
        frames.append(g)
    return pd.concat(frames)


def load_or_fallback(stations, years, cache_dir="data"):
    """Вернуть (DataFrame, источник). Источник: 'IEM' или 'SYNTHETIC_FALLBACK'."""
    try:
        return load_stations(stations, years, cache_dir), "IEM"
    except Exception as e:  # нет сети / формат изменился
        print(f"[!] IEM недоступен ({type(e).__name__}: {e}). Использую СИНТЕТИЧЕСКИЙ запасной ряд.")
        return synthetic_fallback(stations, years), "SYNTHETIC_FALLBACK"
