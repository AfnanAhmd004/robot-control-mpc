"""Discrete-time infinite-horizon LQR."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


def dlqr(A, B, Q, R):
    """Return gain K (u = -K x) and the Riccati solution P."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P
