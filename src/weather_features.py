"""Признаки для угроз 5, 7, 8, 9: видео-осадки, визуальная навигация (оптический поток), фильтры LiDAR, ударные события."""
import numpy as np
import cv2
from scipy.ndimage import label, gaussian_filter
from scipy.spatial import cKDTree

VX = 1.0


def _gray(frames):
    return frames.mean(axis=-1)


def compensated_diff(frames):
    """D_t = I_t(x) − I_{t−1}(x+VX): неподвижный в мире фон сокращается (движение известно по телеметрии)."""
    g = _gray(frames)
    s = int(round(VX))
    return g[1:, :, :-s] - g[:-1, :, s:]


def video_features(frames, thr=0.09, min_area=2):
    D = compensated_diff(frames)
    n = D.shape[0]
    cnt, areas, elongs, vert = [], [], [], []
    for t in range(n):
        lab, k = label(np.abs(D[t]) > thr, structure=np.ones((3, 3)))
        c = 0
        for i in range(1, k + 1):
            ys, xs = np.where(lab == i)
            a = len(ys)
            if a < min_area or a > 400:
                continue
            c += 1; areas.append(a)
            if a >= 4:
                ev, evec = np.linalg.eigh(np.cov(np.vstack([ys, xs])))
                elongs.append(float(np.sqrt(ev[-1] / max(ev[0], 1e-3))))
                vert.append(float(abs(evec[0, -1])))        # косинус угла главной оси с вертикалью
        cnt.append(c)
    g = _gray(frames)
    gy, gx = np.gradient(g[0])
    total = float((D ** 2).mean(axis=(1, 2)).mean())
    persist = float((D.mean(axis=0) ** 2).mean() / (total + 1e-9))
    return dict(n_blobs=float(np.mean(cnt)), area_med=float(np.median(areas)) if areas else 0.0,
                area_p90=float(np.percentile(areas, 90)) if areas else 0.0,
                elong_med=float(np.median(elongs)) if elongs else 1.0, vert_med=float(np.median(vert)) if vert else 0.5,
                res_energy=total, persist=persist, grad=float(np.hypot(gx, gy).mean()),
                contrast=float(g.std(axis=(1, 2)).mean()), mean=float(g.mean()))


def vio_inliers(frames, max_pairs=8):
    """Прокси фронтенда VIO: детектор углов + LK-трекинг вперёд-назад; инлайер - ошибка FB < 0.5 px и поток ≈ известному сдвигу."""
    g = (_gray(frames) * 255).astype(np.uint8)
    inl, tot = [], []
    for t in range(0, min(len(g) - 1, max_pairs)):
        p0 = cv2.goodFeaturesToTrack(g[t], 150, 0.01, 6)
        if p0 is None or len(p0) < 5:
            inl.append(0); tot.append(0); continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g[t], g[t + 1], p0, None, winSize=(15, 15), maxLevel=2)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g[t + 1], g[t], p1, None, winSize=(15, 15), maxLevel=2)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        flow = (p1 - p0).reshape(-1, 2)
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < 0.5) & (np.linalg.norm(flow - np.array([-VX, 0.0]), axis=1) < 0.7)
        inl.append(int(ok.sum())); tot.append(len(p0))
    return float(np.mean(inl)), float(np.mean(tot))


# ---------------- LiDAR ----------------
def dror(xyz, R, d_ang=np.radians(0.4), k=3.0, r_min=0.06, n_min=3, r_apply=30.0):
    """Dynamic Radius Outlier Removal (Charron et al., 2018): радиус поиска растёт с дальностью ∝ R·Δθ.
    Применяется только в ближней зоне (R < r_apply), где живут помехи осадков. Возвращает mask (True = оставить)."""
    keep = np.ones(len(R), bool)
    near = R < r_apply
    if near.sum() == 0:
        return keep
    tree = cKDTree(xyz)
    rad = np.maximum(r_min, k * R * d_ang)
    cnt = np.array(tree.query_ball_point(xyz[near], rad[near], return_length=True)) - 1
    keep[near] = cnt >= n_min
    return keep


def ror(xyz, R, radius=0.5, n_min=3, r_apply=30.0):
    keep = np.ones(len(R), bool); near = R < r_apply
    if near.sum() == 0:
        return keep
    tree = cKDTree(xyz)
    cnt = np.array(tree.query_ball_point(xyz[near], radius, return_length=True)) - 1
    keep[near] = cnt >= n_min
    return keep


def sor(xyz, R, k=8, std_mult=1.0, r_apply=30.0):
    keep = np.ones(len(R), bool); near = np.where(R < r_apply)[0]
    if len(near) < k + 1:
        return keep
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz[near], k=k + 1)
    md = dist[:, 1:].mean(1)
    keep[near] = md <= md.mean() + std_mult * md.std()
    return keep


def filter_metrics(scan, keep):
    c, o = scan["clutter"], scan["obstacle"]
    removed = ~keep
    return dict(clutter_recall=float(removed[c].mean()) if c.any() else np.nan,
                clutter_precision=float(c[removed].mean()) if removed.any() else np.nan,
                obstacle_kept=float(keep[o].mean()) if o.any() else np.nan,
                ground_kept=float(keep[~c & ~o].mean()))
