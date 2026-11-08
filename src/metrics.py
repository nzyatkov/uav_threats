"""Метрики оценки: событийные (задержка, ложные тревоги) и калибровка."""
import numpy as np


def first_true_time(alarm, t, t_from=-np.inf, t_to=np.inf):
    """Время первого срабатывания в окне [t_from, t_to] или NaN."""
    alarm = np.asarray(alarm, dtype=bool)
    t = np.asarray(t)
    m = alarm & (t >= t_from) & (t <= t_to)
    return float(t[m][0]) if m.any() else np.nan


def ece(y_true, p, n_bins=10):
    """Expected Calibration Error для бинарной вероятности."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(e)


def reliability_curve(y_true, p, n_bins=10):
    """Точки для reliability diagram: (средняя p, доля положительных, размер бина)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(p[m].mean()); ys.append(y_true[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
