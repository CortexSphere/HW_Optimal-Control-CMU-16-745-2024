import numpy as np
import scipy.linalg
import scipy.sparse as sparse
import osqp
from control import dare, lqr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

# Model parameters
g = 9.81  # m/s^2
m = 1.0  # kg
l = 0.3  # meters
J = 0.2 * m * l ** 2

# Thrust limits
umin = np.array([0.2 * m * g, 0.2 * m * g])
umax = np.array([0.6 * m * g, 0.6 * m * g])

h = 0.05  # time step (20 Hz)


def quad_dynamics(x, u):
    """
    Compute the state derivatives for the planar quadrotor.

    Parameters:
    x : ndarray
        State vector [x, y, theta, x_dot, y_dot, theta_dot].
    u : ndarray
        Control inputs [u1, u2].

    Returns:
    dxdt : ndarray
        State derivative vector.
    """
    theta = x[2]

    x_ddot = (1 / m) * (u[0] + u[1]) * np.sin(theta)
    y_ddot = (1 / m) * (u[0] + u[1]) * np.cos(theta) - g
    theta_ddot = (1 / J) * (l / 2) * (u[1] - u[0])

    dxdt = np.zeros(6)
    dxdt[0:3] = x[3:6]
    dxdt[3] = x_ddot
    dxdt[4] = y_ddot
    dxdt[5] = theta_ddot

    return dxdt


def quad_dynamics_rk4(x, u, h=h):
    """
    Perform Runge-Kutta 4 integration for the quadrotor dynamics.

    Parameters:
    x : ndarray
        Current state vector.
    u : ndarray
        Control input.
    h : float
        Time step.

    Returns:
    x_next : ndarray
        Next state vector after time step h.
    """
    k1 = quad_dynamics(x, u)
    k2 = quad_dynamics(x + 0.5 * h * k1, u)
    k3 = quad_dynamics(x + 0.5 * h * k2, u)
    k4 = quad_dynamics(x + h * k3, u)

    x_next = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def numerical_jacobian(func, x, epsilon=1e-6):
    """
    Compute the Jacobian matrix of a function numerically using finite differences.

    Parameters:
    func : callable
        Function for which to compute the Jacobian.
    x : ndarray
        Point at which to compute the Jacobian.
    epsilon : float
        Perturbation size.

    Returns:
    jac : ndarray
        Jacobian matrix.
    """
    n = x.size
    fx = func(x)
    m = fx.size
    jac = np.zeros((m, n))
    for i in range(n):
        x_perturbed = np.copy(x)
        x_perturbed[i] += epsilon
        fx_perturbed = func(x_perturbed)
        jac[:, i] = (fx_perturbed - fx) / epsilon
    return jac


# Linearized dynamics for hovering
x_hover = np.zeros(6)
u_hover = np.array([0.5 * m * g, 0.5 * m * g])


# Define functions for linearization
def dynamics_rk4_wrapper(x):
    return quad_dynamics_rk4(x, u_hover)


def dynamics_rk4_u_wrapper(u):
    return quad_dynamics_rk4(x_hover, u)


# Compute A and B using numerical Jacobian
A = numerical_jacobian(dynamics_rk4_wrapper, x_hover)
B = numerical_jacobian(dynamics_rk4_u_wrapper, u_hover)

# Number of states and controls
Nx = 6  # number of states
Nu = 2  # number of control inputs

# Final time and number of time steps
Tfinal = 10.0
Nt = int(Tfinal / h) + 1
thist = np.arange(0, h * Nt, h)

# Cost weights
Q = np.eye(Nx)
R = 0.01 * np.eye(Nu)
Qn = np.eye(Nx)


def cost_function(xhist, uhist):
    """
    Compute the total cost over the history of states and controls.

    Parameters:
    xhist : ndarray
        History of states.
    uhist : ndarray
        History of control inputs.

    Returns:
    cost : float
        Total cost.
    """
    cost = 0.5 * xhist[:, -1].T @ Qn @ xhist[:, -1]
    for k in range(xhist.shape[1] - 1):
        cost += 0.5 * xhist[:, k].T @ Q @ xhist[:, k]
        cost += 0.5 * uhist[k].T @ R @ uhist[k]
    return cost


# LQR Hover Controller
# Solve Discrete Algebraic Riccati Equation
P = dare(A, B, Q, R)[0]
# Compute LQR gain
K, _, _ = lqr(A, B, Q, R)


def lqr_controller(t, x, K=K, xref=np.zeros(Nx)):
    """
    LQR controller law.

    Parameters:
    t : float
        Current time (unused).
    x : ndarray
        Current state.
    K : ndarray
        LQR gain matrix.
    xref : ndarray
        Reference state.

    Returns:
    u : ndarray
        Control input.
    """
    return u_hover - K @ (x - xref)


# Build QP matrices for OSQP
Nh = 20  # Horizon length (one second at 20Hz)

# Construct H matrix (quadratic cost)
#H = sparse.block_diag([Q] * Nh + [R] * Nh).tocsc()
H=sparse.block_diag(sparse.kron(sparse.eye(Nh-1),sparse.block_diag([Q]*1+[R]*1))+sparse.block_diag([R]*1+[P]*1)  ).tocsc()
# Initialize b and other matrices
b = np.zeros(Nh * (Nx + Nu))

# Construct C matrix (dynamics constraints)
# First, state constraints
C_top = sparse.hstack([B, -sparse.eye(Nx)])
# Then, the rest of the dynamics
C_rest = sparse.kron(sparse.eye(Nh - 1), sparse.hstack([A, B]))
C = sparse.vstack([C_top, C_rest]).tocsc()

# Thrust constraints
U = sparse.kron(sparse.eye(Nh), np.hstack([np.eye(Nu), np.zeros((Nu, Nx))]))
Theta = sparse.kron(sparse.eye(Nh), np.array([[0, 0, 0, 0, 1, 0, 0, 0]]))

# Dynamics + thrust limit + theta constraints
D = sparse.vstack([
    C,
    U,
    sparse.kron(sparse.eye(Nh), np.array([[1, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 0]])),
    sparse.kron(sparse.eye(Nh), np.array([[0, 0, 1, 0, 0, 0]]))
]).tocsc()

# Define lower and upper bounds
lb = np.concatenate([
    np.zeros(Nx * Nh),  # Dynamics equality constraints
    np.tile(umin - u_hover, Nh),  # Control lower limits
    np.tile(-0.2, Nh)  # Theta lower limits
])

ub = np.concatenate([
    np.zeros(Nx * Nh),  # Dynamics equality constraints
    np.tile(umax - u_hover, Nh),  # Control upper limits
    np.tile(0.2, Nh)  # Theta upper limits
])

# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P=H, q=b, A=D, l=lb, u=ub, verbose=False, eps_abs=1e-8, eps_rel=1e-8, polish=True)


def mpc_controller(t, x, K=K, xref=np.zeros(Nx)):
    """
    MPC controller using OSQP.

    Parameters:
    t : float
        Current time.
    x : ndarray
        Current state.
    K : ndarray
        LQR gain matrix.
    xref : ndarray
        Reference state.

    Returns:
    u : ndarray
        Control input.
    """
    # Update the first Nx constraints to be the current state
    prob.update(l=lb, u=ub)

    # Update the QP's b vector based on the current state and reference
    # This is a placeholder as detailed implementation requires proper mapping
    # Here, we assume a simplified scenario

    # Solve QP
    results = prob.solve()

    if results.info.status != 'solved':
        print("OSQP did not find a solution!")
        return u_hover  # Fallback to hover

    # Extract the first control input
    delta_u = results.x[:Nu]

    return u_hover + delta_u


def closed_loop(x0, controller, Nt, xref=np.zeros(Nx)):
    """
    Simulate closed-loop system.

    Parameters:
    x0 : ndarray
        Initial state.
    controller : callable
        Controller function.
    Nt : int
        Number of time steps.
    xref : ndarray
        Reference state.

    Returns:
    xhist : ndarray
        History of states.
    uhist : ndarray
        History of control inputs.
    """
    xhist = np.zeros((Nx, Nt))
    uhist = np.zeros((Nu, Nt - 1))
    xhist[:, 0] = x0
    for k in range(Nt - 1):
        u = controller(k * h, xhist[:, k], xref)
        # Enforce control limits
        u = np.clip(u, umin, umax)
        uhist[:, k] = u
        # Integrate dynamics
        x_next = quad_dynamics_rk4(xhist[:, k], u)
        xhist[:, k + 1] = x_next
    return xhist, uhist


# Define reference and initial states
x_ref = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
x0 = np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0])

# Simulate closed-loop with LQR controller
xhist1, uhist1 = closed_loop(x0, lqr_controller, Nt, x_ref)

# Simulate closed-loop with MPC controller
xhist2, uhist2 = closed_loop(x0, mpc_controller, Nt, x_ref)

# Plotting States and Controls
import matplotlib.pyplot as plt

time = thist[:Nt]

plt.figure(figsize=(12, 8))

# Plot states for LQR
plt.subplot(2, 2, 1)
plt.plot(time, xhist1[0, :], label='x')
plt.plot(time, xhist1[1, :], label='y')
plt.plot(time, xhist1[2, :], label='theta')
plt.title('LQR Controller States')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.grid(True)

# Plot controls for LQR
plt.subplot(2, 2, 2)
plt.step(time[:-1], uhist1[0, :], where='post', label='u1')
plt.step(time[:-1], uhist1[1, :], where='post', label='u2')
plt.title('LQR Controller Inputs')
plt.xlabel('Time [s]')
plt.ylabel('Control Inputs')
plt.legend()
plt.grid(True)

# Plot states for MPC
plt.subplot(2, 2, 3)
plt.plot(time, xhist2[0, :], label='x')
plt.plot(time, xhist2[1, :], label='y')
plt.plot(time, xhist2[2, :], label='theta')
plt.title('MPC Controller States')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.grid(True)

# Plot controls for MPC
plt.subplot(2, 2, 4)
plt.step(time[:-1], uhist2[0, :], where='post', label='u1')
plt.step(time[:-1], uhist2[1, :], where='post', label='u2')
plt.title('MPC Controller Inputs')
plt.xlabel('Time [s]')
plt.ylabel('Control Inputs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Dynamic Visualization using Matplotlib Animation
def animate_quadrator(xhist, controller_type="Controller"):
    """
    Animate the quadrotor's movement based on its state history.

    Parameters:
    xhist : ndarray
        History of states.
    controller_type : str
        Type of controller (for title purposes).
    """
    # Extract x and y positions
    xs = xhist[0, :]
    ys = xhist[1, :]
    thetas = xhist[2, :]

    # Determine plot limits
    margin = 1
    x_min, x_max = np.min(xs) - margin, np.max(xs) + margin
    y_min, y_max = np.min(ys) - margin, np.max(ys) + margin

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Quadrotor Movement using {controller_type}")
    ax.axhline(y=0, color="r", linestyle="--", label="Ground level")
    ax.legend()
    ax.grid()

    # Create the quadrotor representation (as a triangle to show orientation)
    quad_size = 0.1  # Size of the quadrotor for visualization
    quad, = ax.plot([], [], "bo", markersize=8, label="Quadrotor")
    trajectory, = ax.plot([], [], "b-", lw=1, label="Trajectory")
    orientation, = ax.plot([], [], "k-", lw=2, label="Orientation")

    ax.legend()

    # Initialize the animation
    def init():
        quad.set_data([], [])
        trajectory.set_data([], [])
        orientation.set_data([], [])
        return quad, trajectory, orientation

    # Update the animation
    def update(frame):
        # Update quadrotor position
        quad.set_data(xs[frame], ys[frame])

        # Update trajectory
        trajectory.set_data(xs[:frame + 1], ys[:frame + 1])

        # Update orientation
        theta = thetas[frame]
        # Calculate orientation line
        ori_x = xs[frame] + quad_size * np.cos(theta)
        ori_y = ys[frame] + quad_size * np.sin(theta)
        orientation.set_data([xs[frame], ori_x], [ys[frame], ori_y])

        return quad, trajectory, orientation

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=range(len(xs)), init_func=init,
        interval=h * 1000, blit=True
    )

    # Save the animation to a file (optional)
    anim.save(f"quadrotor_{controller_type.lower()}_simulation.gif", writer="pillow", fps=int(1 / h))
    print(f"The GIF graph of the {controller_type} simulation has been saved successfully!")

    plt.show()


# Animate LQR Controller Simulation
animate_quadrator(xhist1, controller_type="LQR")

# Animate MPC Controller Simulation
animate_quadrator(xhist2, controller_type="MPC")
