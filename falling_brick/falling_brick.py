import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

def cost(qp, x):
    """
    Calculate the cost function for the QP.
    """
    return 0.5 * x.T @ qp['Q'] @ x + qp['q'].T @ x


def c_eq(qp, x):
    """
    Calculate the equality constraints.
    """
    return qp['A'] @ x - qp['b']


def h_ineq(qp, x):
    """
    Calculate the inequality constraints.
    """
    return qp['G'] @ x - qp['h']


def kkt_conditions(qp, x, lam, mu):
    """
    Calculate the KKT conditions for the QP.
    """
    return np.hstack([
        qp['Q'] @ x + qp['q'] + qp['A'].T @ lam + qp['G'].T @ mu,  # Stationarity
        c_eq(qp, x),  # Equality constraints
        h_ineq(qp, x) * mu  # Complementarity condition
    ])


def mask_matrix(qp, x, mu, rho):
    """
    Create the mask matrix for the augmented Lagrangian.
    """
    h = h_ineq(qp, x)
    I_rho = np.zeros((len(mu), len(mu)))
    for i in range(len(mu)):
        if h[i] < 0 and mu[i] == 0:
            I_rho[i, i] = 0
        else:
            I_rho[i, i] = rho
    return I_rho


def augmented_lagrangian(qp, x, lam, mu, rho):
    """
    Compute the augmented Lagrangian function.
    """
    h = h_ineq(qp, x)
    c = c_eq(qp, x)
    I_rho = mask_matrix(qp, x, mu, rho)
    return (
        cost(qp, x)
        + lam.T @ c
        + mu.T @ h
        + 0.5 * rho * np.sum(c**2)
        + 0.5 * h.T @ I_rho @ h
    )


def solve_qp(qp, verbose=True, max_iters=100, tol=1e-8):
    """
    Solve the QP using the augmented Lagrangian method.
    """
    x = np.zeros(len(qp['q']))
    lam = np.zeros(len(qp['b']))
    mu = np.zeros(len(qp['h']))
    rho = 1.0
    phi = 10.0

    if verbose:
        print("iter   |grad_L_x| |grad_AL_x| max(h)     |c|        compl     rho")
        print("----------------------------------------------------------------")

    for main_iter in range(max_iters):
        # Minimize the augmented Lagrangian
        def lagrangian(_x):
            return augmented_lagrangian(qp, _x, lam, mu, rho)

        def lagrangian_grad(_x):
            # Numerical gradient for the augmented Lagrangian
            epsilon = 1e-8
            grad = np.zeros_like(_x)
            for i in range(len(_x)):
                dx = np.zeros_like(_x)
                dx[i] = epsilon
                grad[i] = (lagrangian(_x + dx) - lagrangian(_x - dx)) / (2 * epsilon)
            return grad

        res = minimize(
            lagrangian,
            x,
            method="BFGS",
            jac=lagrangian_grad,
            options={"gtol": tol, "disp": False},
        )

        x = res.x

        # Update Lagrange multipliers and penalty term
        if len(qp["b"]) > 0:  # Only update lambda if there are equality constraints
            lam += rho * c_eq(qp, x)
        if len(qp["h"]) > 0:  # Only update mu if there are inequality constraints
            mu = np.maximum(0, mu + rho * h_ineq(qp, x))
        rho *= phi

        if verbose:
            grad_l = lagrangian_grad(x)
            max_h = np.max(h_ineq(qp, x)) if len(qp["h"]) > 0 else 0.0
            norm_c = np.linalg.norm(c_eq(qp, x), np.inf) if len(qp["b"]) > 0 else 0.0
            compl = abs(np.dot(mu, h_ineq(qp, x))) if len(qp["h"]) > 0 else 0.0
            print(
                f"{main_iter:3d}  {np.linalg.norm(grad_l):7.2e}  {max_h:7.2e}  "
                f"{norm_c:7.2e}  {compl:7.2e}  {rho:5.0e}"
            )

        # Convergence check
        if (
            (len(qp["h"]) == 0 or np.max(h_ineq(qp, x)) < tol)
            and (len(qp["b"]) == 0 or np.linalg.norm(c_eq(qp, x), np.inf) < tol)
        ):
            if verbose:
                print("Converged successfully!")
            return x, lam, mu

    raise ValueError("QP solver failed to converge")






def brick_simulation_qp(q, v, mass=1.0, delta_t=0.01):
    Q = np.array([[mass, 0], [0, mass]])
    q_vec = mass * (delta_t * np.array([0, 9.81]) - v)
    G = np.array([[0, -delta_t]])
    h = np.dot(np.array([[0, 1]]), q)
    return Q, q_vec, G, h


def augmented_lagrangian_qp(Q, q_vec, G, h, max_iter=100, tol=1e-6, rho=1.0):
    n = len(q_vec)
    m = len(h)
    # Initialize variables
    v = np.zeros(n)  # primal variable
    lambda_ = np.zeros(m)  # dual variable (Lagrange multipliers)

    for it in range(max_iter):
        # Solve the primal problem (minimize augmented Lagrangian)
        def lagrangian(v):
            penalty = 0.5 * rho * np.linalg.norm(np.maximum(0, G @ v - h))**2
            dual = lambda_ @ (G @ v - h)
            return 0.5 * v.T @ Q @ v + q_vec @ v + dual + penalty

        # Gradient of the Lagrangian
        def grad_lagrangian(v):
            return Q @ v + q_vec + (lambda_+rho*( G.dot(v)-h )) * G.T

        # Simple gradient descent
        for _ in range(100):
            grad = grad_lagrangian(v)
            step_size = 1e-3  # Small step size for stability
            v = v - step_size * grad

        # Update Lagrange multipliers
        residual = G @ v - h
        lambda_ = np.maximum(0, lambda_ + rho * residual)

        # Check convergence
        if np.linalg.norm(residual) < tol and np.linalg.norm(grad) < tol:
            break

    return v, lambda_


def simulate_brick_with_visualization():
    """
    Simulate the falling brick using the QP formulation and visualize results with animation.
    """
    dt = 0.01
    T = 3.0
    t_vec = np.arange(0, T + dt, dt)
    N = len(t_vec)

    # Initialize positions and velocities
    qs = np.zeros((N, 2))  # Shape (N, 2) for positions
    vs = np.zeros((N, 2))  # Shape (N, 2) for velocities

    qs[0] = np.array([0, 1.0])  # Initial position
    vs[0] = np.array([1, 4.5])  # Initial velocity

    for k in range(N - 1):
        Q, q_vec, G, h = brick_simulation_qp(qs[k], vs[k], delta_t=dt)
        qp = {
            "Q": Q,
            "q": q_vec,
            "A": np.zeros((0, 2)),  # No equality constraints
            "b": np.zeros(0),
            "G": G,
            "h": h,
        }
        vs[k + 1], _, _ = solve_qp(qp, verbose=True, tol=1e-8)
        qs[k + 1] = qs[k] + dt * vs[k + 1]

    # Extract x and y coordinates as sequences
    xs = qs[:, 0]
    ys = qs[:, 1]

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Brick Falling and Sliding on Ice")
    ax.axhline(y=0, color="r", linestyle="--", label="Ground level")
    ax.legend()
    ax.grid()

    # Create the brick (as a point) and the trajectory line
    brick, = ax.plot([], [], "bo", markersize=8, label="Brick")
    trajectory, = ax.plot([], [], "b-", lw=1)

    # Initialize the animation
    def init():
        brick.set_data([], [])
        trajectory.set_data([], [])
        return brick, trajectory

    # Update the animation
    def update(frame):
        brick.set_data([xs[frame]], [ys[frame]])  # Pass as sequences ([x], [y])
        trajectory.set_data(xs[:frame + 1], ys[:frame + 1])  # Update trajectory
        return brick, trajectory

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=N, init_func=init, interval=dt * 1000, blit=True
    )

    # Save the animation to a file (optional)
    anim.save("brick_simulation.gif", writer="pillow", fps=30)
    print("the GIF graph of whole process has been saved successfully!")



if __name__ == "__main__":
    simulate_brick_with_visualization()
