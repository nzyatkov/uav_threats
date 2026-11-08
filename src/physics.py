"""Физические формулы для метео-признаков (БВС-5: temperature, humidity).

Все температуры в °C, влажность в %.
"""
import numpy as np

# Константы Magnus (Alduchov & Eskridge, 1996), диапазон -45..60 °C
_A, _B = 17.625, 243.04


def dewpoint(T, RH):
    """Точка росы по формуле Magnus."""
    T = np.asarray(T, dtype=float)
    RH = np.clip(np.asarray(RH, dtype=float), 0.1, 100.0)
    g = np.log(RH / 100.0) + _A * T / (_B + T)
    return _B * g / (_A - g)


def rel_humidity(T, Td):
    """Относительная влажность (%) по T и точке росы."""
    T = np.asarray(T, dtype=float)
    Td = np.asarray(Td, dtype=float)
    return 100.0 * np.exp(_A * Td / (_B + Td) - _A * T / (_B + T))


def wet_bulb_stull(T, RH):
    """Температура мокрого термометра (Stull, 2011).

    Применимо при RH 5..99 %, T -20..50 °C при давлении ~1000 гПа.
    """
    T = np.asarray(T, dtype=float)
    RH = np.clip(np.asarray(RH, dtype=float), 5.0, 99.0)
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


def air_at_altitude(T0, Td0, dz, lapse_T=0.0065, lapse_Td=0.002):
    """Грубая стандартная атмосфера: T и Td на высоте dz (м) над точкой, где измерены T0, Td0.

    T падает ~6.5 К/км, Td ~2 К/км (при постоянном отношении смеси).
    Td не может превышать T (иначе это облако): ограничиваем сверху.
    """
    T = np.asarray(T0, dtype=float) - lapse_T * np.asarray(dz, dtype=float)
    Td = np.asarray(Td0, dtype=float) - lapse_Td * np.asarray(dz, dtype=float)
    return T, np.minimum(Td, T - 0.05)
