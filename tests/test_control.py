import numpy as np

from control import LTVMPC, CartPole, Unicycle, dlqr, figure_eight


def test_dare_solution_satisfies_riccati():
    A, B = CartPole().linearize(0.02)
    Q, R = np.diag([1, 1, 10, 1]), np.array([[0.1]])
    K, P = dlqr(A, B, Q, R)
    rhs = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
    assert np.allclose(P, rhs, atol=1e-6)
    assert np.all(np.abs(np.linalg.eigvals(A - B @ K)) < 1)  # closed loop is stable


def test_lqr_balances_cartpole():
    cp = CartPole()
    A, B = cp.linearize(0.02)
    K, _ = dlqr(A, B, np.diag([1, 1, 50, 5]), np.array([[0.1]]))
    x = np.array([0, 0, 0.2, 0])
    for _ in range(400):
        x = cp.step(x, float(np.clip(-(K @ x)[0], -20, 20)), 0.02)
    assert abs(x[2]) < 1e-3 and abs(x[0]) < 0.05


def test_unicycle_jacobian():
    x, u, dt = np.array([1.0, 2.0, 0.7]), np.array([1.5, 0.3]), 0.1
    A, B = Unicycle.linearize(x, u, dt)
    num_A = np.column_stack([(Unicycle.step(x + e, u, dt) - Unicycle.step(x - e, u, dt)) / 2e-6 for e in np.eye(3) * 1e-6])
    assert np.allclose(A, num_A, atol=1e-6)


def test_mpc_tracks_and_respects_bounds():
    dt = 0.1
    x_ref, u_ref = figure_eight(T=20, dt=dt)
    mpc = LTVMPC(Unicycle, 10, dt, np.diag([10, 10, 1]), np.diag([0.1, 0.1]), [0.0, -1.5], [2.0, 1.5])
    x = np.array([0.4, -0.5, 0.2])
    us, errs = [], []
    for k in range(len(u_ref) - mpc.N):
        u = mpc.solve(x, x_ref[k : k + mpc.N + 1], u_ref[k : k + mpc.N])
        x = Unicycle.step(x, u, dt)
        us.append(u); errs.append(np.linalg.norm(x[:2] - x_ref[k + 1, :2]))
    us = np.array(us)
    assert np.mean(errs[50:]) < 0.1
    assert us[:, 0].min() >= -1e-9 and us[:, 0].max() <= 2 + 1e-9 and np.abs(us[:, 1]).max() <= 1.5 + 1e-9
