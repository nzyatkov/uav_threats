"""Детекторы лазерного/прожекторного воздействия (угроза 10) и кандидатов «пожара» в RGB (связь с угрозой 16)."""
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEAT_ML = ["sat_frac", "amax", "nbl", "mean", "p05", "p99", "log_gain", "sigma", "halo", "Tex_blob",
           "d_sat_frac", "d_log_gain", "d_p05", "d_sigma", "d_mean", "n_down12"]


def rule_laser(F, jump_thr=0.001, dT_gate=8.0, halo_thr=None, hold=6, use_thermal=True,
               day=False, day_factor=1.0, warmup=15, k_sd=8.0, pulse_confirm=False, big_jump=0.05):
    """Zero-shot правило: резкий прирост доли насыщенных пикселей относительно недавнего прошлого,
    за вычетом очагов (тепловая аномалия в зоне засветки), опционально - требование ореола.
    День (по высоте Солнца из telemetry) повышает порог в day_factor раз: контекстный приор против солнечных бликов.
    База замораживается во время тревоги. Возвращает (alarm, score)."""
    sat = F["sat_frac"].values
    n = len(sat)
    alarm = np.zeros(n, bool); score = np.zeros(n); cand = np.zeros(n, bool)
    hist = []                      # значения sat_frac вне тревоги
    for t in range(n):
        base = np.median(hist[-25:-3]) if len(hist) > 6 else sat[:max(t, 1)].mean()
        jump = sat[max(0, t - 2):t + 1].max() - base
        score[t] = jump
        hh = np.asarray(hist[-25:])
        sd = 1.4826 * np.median(np.abs(hh - np.median(hh))) if len(hh) > 6 else 0.0
        c = (jump > jump_thr * (day_factor if day else 1.0) + k_sd * sd) and t >= warmup
        if use_thermal:
            c &= F["Tex_blob_max3"].values[t] < dT_gate
        if pulse_confirm:       # импульсный режим: нужен хотя бы один спад; очень большие засветки пропускаем сразу
            c &= (F["n_down12"].values[t] >= 1) or (jump > big_jump)
        if halo_thr is not None:
            c &= F["halo_max3"].values[t] > halo_thr
        cand[t] = c
        alarm[t] = cand[max(0, t - hold + 1):t + 1].any()
        if not alarm[t]:
            hist.append(sat[t])
    return alarm, score


class AnomalyModels:
    """Одноклассовые модели по признакам кадра: Isolation Forest и PCA (линейный автоэнкодер)."""
    def __init__(self, seed=0, n_comp=5):
        self.sc = StandardScaler(); self.seed = seed
        self.iforest = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
        self.pca = PCA(n_components=n_comp, random_state=seed)

    def fit(self, Fn):
        X = self.sc.fit_transform(Fn[FEAT_ML].values)
        self.iforest.fit(X); self.pca.fit(X)
        self.thr = {}
        for k in ("iforest", "pca"):
            self.thr[k] = np.quantile(self._score(k, X), 0.995)
        return self

    def _score(self, kind, X):
        if kind == "iforest":
            return -self.iforest.score_samples(X)
        R = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - R) ** 2).sum(1)

    def scores(self, F):
        X = self.sc.transform(F[FEAT_ML].values)
        return {k: self._score(k, X) for k in ("iforest", "pca")}

    def alarm(self, F, kind, q=1.0, hold=6):
        s = self.scores(F)[kind]
        c = s > self.thr[kind] * q
        out = np.array([c[max(0, t - hold + 1):t + 1].any() for t in range(len(c))])
        return out, s


def seq_event_outcome(alarm, lab_laser, onset, win=20):
    """Для последовательности с лазером: обнаружение в окне [onset, onset+win), задержка в кадрах."""
    idx = np.where(alarm[onset:onset + win])[0]
    return (len(idx) > 0), (int(idx[0]) if len(idx) else np.nan)


# ---- кандидаты «пожара» в RGB (применение к угрозе 16) -------------------------

def rgb_fire_candidates(y, T, known_mask=None, amax=400, dT=8.0, mode="rgb"):
    """Число кандидатов в кадре. mode: rgb (насыщенные пятна), thermal_gate (+ тепловое подтверждение),
    prior_mask (RGB, исключая известные источники света), thermal_only (только тепловизор)."""
    Tmed = np.median(T)
    if mode == "thermal_only":
        return int((T - Tmed).max() >= dT)
    m = y >= 0.98
    if mode == "prior_mask" and known_mask is not None:
        m = m & ~known_mask
    lab, n = label(m, structure=np.ones((3, 3)))
    cnt = 0
    for i in range(1, n + 1):
        reg = lab == i
        a = reg.sum()
        if a < 2 or a > amax:
            continue
        if mode == "thermal_gate" and (T[binary_dilation(reg, iterations=3)].max() - Tmed) < dT:
            continue
        cnt += 1
    return cnt


def persistent(cnts, k=3):
    """Тревога, если кандидаты присутствуют k кадров подряд."""
    c = (np.asarray(cnts) > 0).astype(int)
    out = np.zeros(len(c), bool)
    for t in range(k - 1, len(c)):
        out[t] = c[t - k + 1:t + 1].all()
    return out
