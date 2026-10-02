"""Linear time-varying MPC for trajectory tracking with input bounds.

At every step the model is linearised along the reference, the horizon is
condensed into a quadratic program in the input deviations, and the QP is
solved with box constraints. Only the first input is applied (receding horizon).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LTVMPC:
    def __init__(self, model, horizon: int, dt: float, Q, R, u_min, u_max, Qf=None):
        self.model, self.N, self.dt = model, horizon, dt
        self.Q, self.R = np.asarray(Q, float), np.asarray(R, float)
        self.Qf = self.Q if Qf is None else np.asarray(Qf, float)
        self.u_min, self.u_max = np.asarray(u_min, float), np.asarray(u_max, float)
        self._warm = None

    def solve(self, x0, x_ref, u_ref):
        """x_ref: (N+1, nx) reference states; u_ref: (N, nu) reference inputs. Returns the first input."""
        N, nx, nu = self.N, len(x0), u_ref.shape[1]
        As, Bs = zip(*(self.model.linearize(x_ref[k], u_ref[k], self.dt) for k in range(N)))
        # condensed prediction: e_{k} = Phi_k e_0 + sum_j Gamma_{k,j} du_j
        e0 = x0 - x_ref[0]
        e0[2] = wrap(e0[2])
        Phi = [np.eye(nx)]
        for k in range(N):
            Phi.append(As[k] @ Phi[-1])
        Gam = np.zeros((N + 1, N, nx, nu))
        for k in range(1, N + 1):
            for j in range(k):
                M = Bs[j]
                for i in range(j + 1, k):
                    M = As[i] @ M
                Gam[k, j] = M
        Qs = [self.Q] * N + [self.Qf]

        def cost(du_flat):
            du = du_flat.reshape(N, nu)
            J, grad = 0.0, np.zeros((N, nu))
            for k in range(1, N + 1):
                e = Phi[k] @ e0 + np.einsum("jab,jb->a", Gam[k, :k], du[:k])
                Qe = Qs[k] @ e
                J += e @ Qe
                grad[:k] += 2 * np.einsum("jab,a->jb", Gam[k, :k], Qe)
            J += np.einsum("ka,ab,kb->", du, self.R, du)
            grad += 2 * du @ self.R
            return J, grad.ravel()

        bounds = [(lo - ur, hi - ur) for k in range(N) for lo, hi, ur in zip(self.u_min, self.u_max, u_ref[k])]
        x_init = np.zeros(N * nu) if self._warm is None else np.r_[self._warm[nu:], np.zeros(nu)]
        res = minimize(cost, x_init, jac=True, bounds=bounds, method="L-BFGS-B")
        self._warm = res.x
        return u_ref[0] + res.x[:nu]
