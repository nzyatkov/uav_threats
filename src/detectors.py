"""Общие детекторы: гистерезис, CUSUM, лаг-фильтр."""
import numpy as np


def hysteresis(p, on=0.6, off=0.3):
    """Триггер с гистерезисом по вероятности/индексу. Возвращает bool-массив."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    state = False
    for i, v in enumerate(p):
        if not state and v >= on:
            state = True
        elif state and v <= off:
            state = False
        out[i] = state
    return out


def lag_filter(x, tau, dt=1.0, x0=None):
    """Звено первого порядка: dy/dt = (x - y)/tau (точная дискретизация)."""
    x = np.asarray(x, dtype=float)
    a = 1.0 - np.exp(-dt / max(tau, 1e-6))
    y = np.empty_like(x)
    y[0] = x[0] if x0 is None else x0
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def cusum_drop(z, k=0.05, h=0.8):
    """Односторонний CUSUM по положительной величине z (насколько сигнал упал).

    Возвращает (alarm_bool, S). Реагирует на устойчивое превышение k.
    """
    z = np.asarray(z, dtype=float)
    S = np.zeros(len(z))
    alarm = np.zeros(len(z), dtype=bool)
    on = False
    for i in range(1, len(z)):
        S[i] = max(0.0, S[i - 1] + z[i] - k)
        if not on and S[i] > h:
            on = True
        elif on and S[i] < h * 0.25:
            on = False
        alarm[i] = on
    return alarm, S
