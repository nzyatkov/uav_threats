"""Признаки кадра для оптических помех (угрозы 10, 16) и солнечная высота (метаданные: timestamp, lat, lon из БВС-1)."""
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation, convolve

_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)   # маска Иммеркера
SAT = 0.98


def noise_sigma(y, valid=None):
    """Оценка σ шума по Иммеркеру (1996): устойчива к содержимому сцены (слабо реагирует на края)."""
    c = np.abs(convolve(y.astype(float), _K, mode="nearest"))[1:-1, 1:-1]
    if valid is not None:
        valid = valid[1:-1, 1:-1]
        if valid.sum() < 50:
            return np.nan
        c = c[valid]
    return float(np.sqrt(np.pi / 2) * c.mean() / 6)


def frame_features(y, T, gain):
    m = y >= SAT
    lab, n = label(m, structure=np.ones((3, 3)))
    if n:
        areas = np.bincount(lab.ravel())[1:]
        amax, nbl = int(areas.max()), int((areas >= 2).sum())
    else:
        amax, nbl = 0, 0
    mid = ((y >= 0.75) & ~m).sum()
    dil = binary_dilation(m, iterations=3) if n else m
    Tmed = np.median(T)
    return dict(
        sat_frac=float(m.mean()), amax=amax, nbl=nbl, mean=float(y.mean()),
        p05=float(np.percentile(y, 5)), p99=float(np.percentile(y, 99)),
        log_gain=float(np.log(gain)), sigma=noise_sigma(y, ~dil),
        halo=float(mid / (m.sum() + 1.0)),
        Tex_blob=float(T[dil].max() - Tmed) if n else 0.0,
        Tex_glob=float(T.max() - Tmed))


def seq_features(seq, temporal=True):
    """DataFrame признаков по кадрам; temporal добавляет разности относительно прошлого (окно 4..25 кадров назад)."""
    F = pd.DataFrame([frame_features(seq["y"][t], seq["T"][t], seq["gain"][t]) for t in range(len(seq["gain"]))])
    F["sigma"] = F["sigma"].ffill().bfill()
    if temporal:
        for c in ("sat_frac", "log_gain", "p05", "sigma", "mean"):
            ref = F[c].rolling(22, min_periods=5).median().shift(4)
            F[f"d_{c}"] = (F[c] - ref).fillna(0.0)
        # пульсация: число заметных СПАДОВ насыщения за 12 кадров (лазерное шоу мерцает, блик/огни - нет)
        F["n_down12"] = (F["sat_frac"].diff() < -0.0008).astype(int).rolling(12, min_periods=1).sum()
        F["sat_max3"] = F["sat_frac"].rolling(3, min_periods=1).max()
        F["Tex_blob_max3"] = F["Tex_blob"].rolling(3, min_periods=1).max()
        F["halo_max3"] = F["halo"].rolling(3, min_periods=1).max()
    return F


def solar_elevation(ts, lat, lon):
    """Высота Солнца над горизонтом, градусы (упрощённый алгоритм NOAA/Spencer 1971, без рефракции).
    ts: pandas.Timestamp (UTC)."""
    ts = pd.Timestamp(ts)
    doy = ts.dayofyear
    hr = ts.hour + ts.minute / 60 + ts.second / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hr * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return float(90 - np.degrees(np.arccos(np.clip(cz, -1, 1))))
