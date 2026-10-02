"""Reference trajectories."""
from __future__ import annotations

import numpy as np


def figure_eight(T: float = 40.0, dt: float = 0.1, a: float = 5.0):
    """Lemniscate with feed-forward speed and turn rate."""
    t = np.arange(0, T + dt, dt)
    w0 = 2 * np.pi / T
    x, y = a * np.sin(w0 * t), a * np.sin(w0 * t) * np.cos(w0 * t)
    dx, dy = np.gradient(x, dt), np.gradient(y, dt)
    th = np.unwrap(np.arctan2(dy, dx))
    v = np.hypot(dx, dy)
    w = np.gradient(th, dt)
    return np.c_[x, y, th], np.c_[v, w]
