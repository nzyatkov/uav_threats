"""Подавление повторных алертов: состояние с гистерезисом, подтверждением (n подряд) и паузой после отбоя."""


class Gate:
    """update(t, x) -> 'on' | 'off' | None. Включение: x >= on n_on раз подряд и прошла пауза cooldown после последнего отбоя;
    выключение: x <= off n_off раз подряд."""

    def __init__(self, on, off, n_on=1, n_off=1, cooldown=0.0):
        self.on, self.off, self.n_on, self.n_off, self.cooldown = on, off, n_on, n_off, cooldown
        self.reset()

    def reset(self):
        self.active = False; self.c_on = 0; self.c_off = 0; self.t_off = -1e18

    def update(self, t, x):
        if not self.active:
            self.c_on = self.c_on + 1 if x >= self.on else 0
            if self.c_on >= self.n_on and t - self.t_off >= self.cooldown:
                self.active = True; self.c_on = 0; self.c_off = 0
                return "on"
        else:
            self.c_off = self.c_off + 1 if x <= self.off else 0
            if self.c_off >= self.n_off:
                self.active = False; self.c_off = 0; self.t_off = t
                return "off"
        return None
