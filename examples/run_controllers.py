"""LQR balances a cart-pole; LTV-MPC tracks a figure-eight with a differential-drive robot.

    python examples/run_controllers.py --plot   # writes docs/demo.png
"""
import argparse

import numpy as np

from control import LTVMPC, CartPole, Unicycle, dlqr, figure_eight

ap = argparse.ArgumentParser()
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

# ---- LQR cart-pole --------------------------------------------------------------------
dt = 0.02
cp = CartPole()
A, B = cp.linearize(dt)
K, _ = dlqr(A, B, Q=np.diag([1, 1, 50, 5]), R=np.array([[0.1]]))
x = np.array([0.0, 0.0, 0.25, 0.0])  # 0.25 rad (~14 deg) off vertical
xs_cp = [x]
for _ in range(300):
    u = float(np.clip(-(K @ x)[0], -20, 20))
    x = cp.step(x, u, dt)
    xs_cp.append(x)
xs_cp = np.array(xs_cp)
print(f"LQR cart-pole: angle {xs_cp[0, 2]:.2f} rad -> {xs_cp[-1, 2]:.4f} rad after {300 * dt:.0f} s")

# ---- MPC unicycle -----------------------------------------------------------------------
dt = 0.1
x_ref, u_ref = figure_eight(dt=dt)
mpc = LTVMPC(Unicycle, horizon=12, dt=dt, Q=np.diag([10, 10, 1]), R=np.diag([0.1, 0.1]),
             u_min=[0.0, -1.5], u_max=[2.0, 1.5])
x = np.array([0.5, -0.8, 0.3])  # start off the path
traj, inputs = [x], []
for k in range(len(u_ref) - mpc.N):
    u = mpc.solve(x, x_ref[k : k + mpc.N + 1], u_ref[k : k + mpc.N])
    x = Unicycle.step(x, u, dt)
    traj.append(x); inputs.append(u)
traj, inputs = np.array(traj), np.array(inputs)
err = np.linalg.norm(traj[:, :2] - x_ref[: len(traj), :2], axis=1)
print(f"MPC figure-eight: start error {err[0]:.2f} m, mean error after 5 s {err[50:].mean():.3f} m, "
      f"inputs within bounds: {bool((inputs[:, 0] >= -1e-9).all() and (inputs[:, 0] <= 2 + 1e-9).all())}")

if args.plot:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    t = np.arange(len(xs_cp)) * 0.02
    a.plot(t, xs_cp[:, 2], label="pole angle (rad)"); a.plot(t, xs_cp[:, 0], label="cart position (m)")
    a.axhline(0, color="k", lw=0.5); a.set_xlabel("time (s)"); a.legend(); a.set_title("LQR: cart-pole stabilisation")
    b.plot(x_ref[:, 0], x_ref[:, 1], "k--", lw=1, label="reference")
    b.plot(traj[:, 0], traj[:, 1], label="MPC"); b.plot(*traj[0, :2], "ro", label="start")
    b.set_aspect("equal"); b.legend(loc="lower right", fontsize=8); b.set_title("LTV-MPC: figure-eight tracking")
    fig.tight_layout(); fig.savefig("docs/demo.png", dpi=100)
    print("wrote docs/demo.png")
