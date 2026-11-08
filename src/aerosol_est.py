"""Оценка видимости, типа аэрозоля и локальных неоднородностей (шлейф / пылевой фронт)."""
import numpy as np
from scipy.ndimage import minimum_filter, gaussian_filter, label, median_filter, distance_transform_edt
from scipy.optimize import least_squares

LAM = np.array([650.0, 550.0, 470.0])


def dark_channel(I, patch=7):
    return minimum_filter(I.min(axis=2), size=patch)


def estimate_A(I, patch=7, top=0.001):
    """Атмосферная засветка по He et al.: среднее по пикселям с наибольшим тёмным каналом."""
    dc = dark_channel(I, patch)
    n = max(1, int(dc.size * top))
    idx = np.argpartition(dc.ravel(), -n)[-n:]
    A = I.reshape(-1, 3)[idx].mean(0)
    return np.clip(A, 0.3, 1.0)


def tau_dcp(I, A=None, patch=7, omega=0.95):
    """Карта оптической толщины −ln t по тёмному каналу."""
    A = estimate_A(I, patch) if A is None else A
    t = 1.0 - omega * dark_channel(I / A, patch)
    return -np.log(np.clip(t, 0.05, 1.0)), A


def beta_from_rows(tau_map, d, intercept=True, tau_sat=2.2):
    """β̂ из строкового профиля τ̃(y) ≈ a + β d (intercept) либо β d (нулевое смещение, классический DCP).
    Строки с насыщенной оценкой (τ̃ ≥ tau_sat: t̃ упирается в нижнюю границу 0,05) исключаются."""
    row = np.median(tau_map, axis=1)
    m = row < tau_sat
    if m.sum() < 8:
        m = np.ones_like(m)
    dd, rr = d[m], row[m]
    if intercept:
        b, a = np.polyfit(dd, rr, 1)
    else:
        b = float((rr * dd).sum() / (dd * dd).sum())
    return max(float(b), 1e-7)


def beta_lidar(R, I, alpha_nom=1.3):
    """β_550 по наклону ln(I R²) против R (закон Бугера-Ламберта, двойной путь)."""
    if len(R) < 50:
        return np.nan
    s, _ = np.polyfit(R, np.log(I * R ** 2), 1)
    bL = max(-s / 2.0, 1e-7)
    return bL / (550.0 / 905.0) ** alpha_nom


def fit_channel_beta(profile, d, A_c, beta0):
    """Подгонка Ī(y) = A + (J − A) exp(−β d) по строкам; J, β свободны."""
    def res(p):
        J, b = p
        return A_c + (J - A_c) * np.exp(-b * d) - profile
    sol = least_squares(res, x0=[float(np.median(profile)), max(beta0, 1e-5)],
                        bounds=([0.0, 1e-7], [1.0, 5e-2]))
    return float(sol.x[1])


def fit_thermal_beta(Tprofile, d, T_air, beta0):
    def res(p):
        Tg, b = p
        return T_air + (Tg - T_air) * np.exp(-b * d) - Tprofile
    sol = least_squares(res, x0=[float(np.median(Tprofile)), max(beta0, 1e-6)],
                        bounds=([T_air - 30, 1e-8], [T_air + 30, 5e-2]))
    return float(sol.x[1])


def spectral_features(I, T, d, t_air):
    """β̂_vis (DCP+глубина), по каналам, β̂_LWIR, κ̂ = β_T/β_550, α̂ - всё при известной геометрии."""
    tau, A = tau_dcp(I)
    b_dcp = beta_from_rows(tau, d, intercept=True)
    prof = np.median(I, axis=1)                     # (H,3)
    b_c = [fit_channel_beta(prof[:, c], d, A[c], b_dcp * (550.0 / LAM[c]) ** 1.3) for c in range(3)]
    b_T = fit_thermal_beta(np.median(T, axis=1), d, t_air, 0.03 * b_dcp)
    alpha = np.log(b_c[2] / b_c[0]) / np.log(LAM[0] / LAM[2])
    return dict(beta_dcp=b_dcp, beta_R=b_c[0], beta_G=b_c[1], beta_B=b_c[2], beta_T=b_T,
                kappa=b_T / b_c[1], alpha=float(alpha), tau_max=b_dcp * float(d.max()))


def vis_features(I, T, d):
    """Табличные признаки для регрессии ln V (без знания глубины в явном виде кроме профиля)."""
    gray = I.mean(2)
    tau, A = tau_dcp(I)
    b1 = beta_from_rows(tau, d, intercept=True); b0 = beta_from_rows(tau, d, intercept=False)
    gy, gx = np.gradient(gray)
    mx, mn = I.max(2), I.min(2)
    row = np.median(tau, 1)
    return dict(b_slope=b1, b_zero=b0, tau_mean=float(tau.mean()), tau_top=float(row[:10].mean()), tau_bot=float(row[-10:].mean()),
                contrast=float(gray.std()), grad=float(np.hypot(gx, gy).mean() / (gray.mean() + 1e-3)),
                sat=float(((mx - mn) / (mx + 1e-6)).mean()), dc=float(dark_channel(I).mean()),
                rb=float((I[..., 0] - I[..., 2]).mean()), Tstd=float(T.std()), gmean=float(gray.mean()))


# ---------------- локальные неоднородности ----------------

def anomaly_analysis(I, T, d, thr=0.25, min_area_frac=0.03, hot_dT=8.0, hot_dist=8.0, I_ref=None, ref_diff=False):
    """Компонента аномального ослабления (шлейф/фронт) и тепловые пятна (очаги/ложные)."""
    tau, A = tau_dcp(I)
    if I_ref is not None and ref_diff:                # разность с эталоном в цветовом пространстве (устойчива к яркой местности)
        tau = 6.0 * np.abs(I - I_ref).mean(axis=2)
    elif I_ref is not None:                          # эталонный пролёт: убираем статический вклад местности (по тёмному каналу)
        tau = tau - tau_dcp(I_ref)[0]
    tau = gaussian_filter(tau, 2.0)
    ref = np.percentile(tau, 25, axis=1, keepdims=True)           # фон по строке (шлейф/фронт поднимают τ)
    resid = tau - ref
    lab, n = label(resid > thr, structure=np.ones((3, 3)))
    comp = None
    if n:
        areas = np.bincount(lab.ravel())[1:]
        k = int(areas.argmax()) + 1
        if areas[k - 1] >= min_area_frac * I.shape[0] * I.shape[1]:
            comp = lab == k
    out = dict(comp_area=0.0, bbox_h=0.0, bbox_w=0.0, elong=1.0, comp_mean=0.0, comp_max=float(resid.max()))
    cpos = None
    if comp is not None:
        ys, xs = np.where(comp)
        out.update(comp_area=comp.mean(), bbox_h=(np.ptp(ys) + 1) / I.shape[0], bbox_w=(np.ptp(xs) + 1) / I.shape[1],
                   comp_mean=float(resid[comp].mean()))
        ev = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs]))) if len(ys) > 3 else np.array([1, 1])
        out["elong"] = float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))
        j = np.argmax(np.where(comp, resid, -9)); cpos = np.unravel_index(j, comp.shape)
    Tex = T - np.median(T, axis=1, keepdims=True)
    hot = Tex >= hot_dT
    hl, hn = label(hot, structure=np.ones((3, 3)))
    out.update(hot_n=int(hn), hot_max=float(Tex.max()))
    near_root = 0.0
    if hn and comp is not None:
        dist = distance_transform_edt(~comp)               # расстояние от каждого пикселя до компоненты
        near_root = float(dist[hot].min() <= hot_dist)      # очаг лежит у компоненты (у «корня» шлейфа)
    out["hot_near_root"] = near_root
    return out, resid


CLASSES = ["clear", "haze_fine", "dust_coarse", "plume_fire", "plume_hidden", "dust_front", "hot_decoy"]


def classify_rule(an, sp, tau_thr=1.4, kappa_thr=0.12, plume_elong=2.0, comp_min=0.03):
    """Zero-shot логика: неоднородность -> шлейф (вытянутый; очаг у основания => пожар) либо фронт (широкий, без очага);
    иначе тепловое пятно / однородный аэрозоль."""
    if an["comp_area"] > comp_min:
        if an["hot_near_root"] > 0 and an["elong"] >= plume_elong and an["comp_area"] <= 0.5:
            return "plume_fire"
        if an["bbox_w"] >= 0.7 or an["bbox_h"] >= 0.7 or an["comp_area"] > 0.3:
            return "dust_front"
        if an["elong"] >= plume_elong:
            return "plume_hidden"
    if an["hot_n"] > 0:
        return "hot_decoy"
    if sp["tau_max"] > tau_thr:
        return "dust_coarse" if sp["kappa"] > kappa_thr else "haze_fine"
    return "clear"
