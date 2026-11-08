"""Слияние видимости: признаки (БВС-5, НСУ-3, БВС-4 + дальности первого обнаружения) и физические оценки видимости по сенсорам."""
import math
import numpy as np
from .detectors_bvs4 import Bvs4Features, BVS4_FEATS
from . import streams as st

VIS_EXTRA = ["d_first_lidar", "d_first_cam", "d0_radar"]
VIS_FEATS = BVS4_FEATS + VIS_EXTRA


class VisFeatures(Bvs4Features):
    """Bvs4Features + абсолютные дальности первого обнаружения lidar/camera и radar на последнем треке препятствия."""

    def feed(self, m):
        x = super().feed(m)
        if x is None:
            return None
        lf = self.last_first
        d_l = lf["lidar"][0] if lf["lidar"] else 150.0; d_c = lf["camera"][0] if lf["camera"] else 120.0
        d0 = (lf["lidar"][1] if lf["lidar"] else 150.0)
        return np.concatenate([x, np.array([d_l, d_c, d0], float)])


def physical_range_estimate(d_l, d_c, age, k_l=0.15, k_c=0.4, max_age=150.0):
    """V по дальности первого обнаружения: lidar V = d/k_l, camera V = d/k_c; насыщение (d у границы) даёт только нижнюю оценку.
    Возвращает (V, надёжно): надёжно - хотя бы один сенсор не в насыщении и трек свежий."""
    if age > max_age:
        return np.nan, False
    est = []
    if 0.0 < d_l < 140.0: est.append(max(d_l, 5.0) / k_l)
    if 0.0 < d_c < 110.0: est.append(max(d_c, 5.0) / k_c)
    if est:
        return float(np.exp(np.mean(np.log(est)))), True
    return 10000.0, False        # оба сенсора в насыщении: видимость не ниже ~1 км (нижняя оценка)
