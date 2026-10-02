"""Robot models: cart-pole and a unicycle (differential-drive) robot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CartPole:
    m_cart: float = 1.0
    m_pole: float = 0.1
    length: float = 0.5  # half-length of the pole
    g: float = 9.81

    def dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """x = [cart pos, cart vel, angle (0 = upright), angular vel]; u = horizontal force."""
        _, v, th, w = x
        mc, mp, l, g = self.m_cart, self.m_pole, self.length, self.g
        s, c = np.sin(th), np.cos(th)
        tmp = (u + mp * l * w**2 * s) / (mc + mp)
        alpha = (g * s - c * tmp) / (l * (4 / 3 - mp * c**2 / (mc + mp)))
        acc = tmp - mp * l * alpha * c / (mc + mp)
        return np.array([v, acc, w, alpha])

    def step(self, x, u, dt):  # RK4
        k1 = self.dynamics(x, u); k2 = self.dynamics(x + dt / 2 * k1, u)
        k3 = self.dynamics(x + dt / 2 * k2, u); k4 = self.dynamics(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def linearize(self, dt: float, eps: float = 1e-5):
        """Discrete-time A, B about the upright equilibrium by finite differences."""
        x0 = np.zeros(4)
        A = np.zeros((4, 4)); B = np.zeros((4, 1))
        for i in range(4):
            e = np.zeros(4); e[i] = eps
            A[:, i] = (self.step(x0 + e, 0.0, dt) - self.step(x0 - e, 0.0, dt)) / (2 * eps)
        B[:, 0] = (self.step(x0, eps, dt) - self.step(x0, -eps, dt)) / (2 * eps)
        return A, B


class Unicycle:
    """x = [px, py, heading]; u = [forward speed v, turn rate w]."""

    @staticmethod
    def step(x, u, dt):
        px, py, th = x
        v, w = u
        return np.array([px + v * np.cos(th) * dt, py + v * np.sin(th) * dt, th + w * dt])

    @staticmethod
    def linearize(x_ref, u_ref, dt):
        th, v = x_ref[2], u_ref[0]
        A = np.array([[1, 0, -v * np.sin(th) * dt], [0, 1, v * np.cos(th) * dt], [0, 0, 1]])
        B = np.array([[np.cos(th) * dt, 0], [np.sin(th) * dt, 0], [0, dt]])
        return A, B
