import numpy as np
import matplotlib.pyplot as plt
import casadi as ca


def dynamic_function(params, x, u):
    """
    计算系统的动力学方程。

    参数：
    - params: 字典，包含系统参数 mc, mp, l
    - x: 状态向量 [p, theta, p_dot, theta_dot]
    - u: 控制输入（标量）

    返回：
    - xdot: 状态导数向量 [p_dot, theta_dot, p_ddot, theta_ddot]
    """
    mc, mp, l = params["mc"], params["mp"], params["l"]
    g = 9.81
    p, theta, p_dot, theta_dot = x[0], x[1], x[2], x[3]
    s = ca.sin(theta)
    c = ca.cos(theta)

    # 定义惯性矩阵 H
    H = ca.vertcat(
        ca.horzcat(mc + mp, mp * l * c),
        ca.horzcat(mp * l * c, mp * l ** 2)
    )

    # 定义科氏力矩阵 C
    C = ca.vertcat(
        ca.horzcat(0, -mp * theta_dot * l * s),
        ca.MX.zeros((1, 2))  # 使用 MX 而非 DM
    )

    # 定义重力向量 G
    G = ca.vertcat(0, mp * g * l * s)

    # 定义输入矩阵 B
    B = ca.vertcat(1, 0)

    # 计算加速度 qdd = [p_ddot, theta_ddot]
    qd = ca.vertcat(p_dot, theta_dot)

    qdd=ca.solve(-H,C@qd+G-B*u)


    # 返回状态导数向量 [p_dot, theta_dot, p_ddot, theta_ddot]
    return ca.vertcat(
        p_dot,
        theta_dot,
        qdd
    )


def dynamic_cons(params, xk, xkp1, uk, ukp1, dt):
    """
    Hermite-Simpson 积分约束函数。

    参数：
    - params: 字典，包含系统参数 mc, mp, l
    - xk: 时间步 k 的状态向量
    - xkp1: 时间步 k+1 的状态向量
    - uk: 时间步 k 的控制输入
    - ukp1: 时间步 k+1 的控制输入
    - dt: 时间步长度

    返回：
    - res: Hermite-Simpson 约束残差向量
    """
    f1 = dynamic_function(params, xk, uk)
    f2 = dynamic_function(params, xkp1, ukp1)

    # 计算中间点状态 x_mid
    x_mid = 0.5 * (xk + xkp1) + (dt / 8) * (f1 - f2)

    # 计算中间点控制 u_mid
    u_mid = 0.5 * (uk + ukp1)

    # 计算中间点导数 f_mid
    f_mid = dynamic_function(params, x_mid, u_mid)

    # Hermite-Simpson 约束
    res = xkp1 - xk - (dt / 6) * (f1 + 4 * f_mid + f2)

    return res


class DIRCOL:
    """
    基于直接区间碰撞法（Direct Collocation）的优化求解器。
    """

    def __init__(self, params, Q, R, Qf, lower_upper_bound_ux, x_start, x_end, Nx, Nu, Nt, dt=0.05):
        self.params = params
        self.x_start = x_start
        self.x_end = x_end
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.lb_u = lower_upper_bound_ux['lb_u']
        self.ub_u = lower_upper_bound_ux['ub_u']
        self.Nx = Nx
        self.Nu = Nu
        self.Nt = Nt
        self.dt = dt

    def solve(self, x_initial, u_initial):
        """
        求解优化问题，返回优化后的状态轨迹和控制轨迹。

        参数：
        - x_initial: 初始状态猜测
        - u_initial: 初始控制猜测

        返回：
        - x_opt: 优化后的状态轨迹
        - u_opt: 优化后的控制轨迹
        """
        opti = ca.Opti()

        # 定义优化变量
        U = opti.variable(self.Nu, self.Nt + 1)  # 控制输入
        X = opti.variable(self.Nx, self.Nt + 1)  # 状态变量

        # 定义目标函数
        obj = 0
        for i in range(self.Nt):
            obj += 0.5 * (X[:, i] - self.x_end).T @ self.Q @ (X[:, i] - self.x_end)
            obj += 0.5 * U[:, i].T @ self.R @ U[:, i]
        obj += 0.5 * (X[:, -1] - self.x_end).T @ self.Qf @ (X[:, -1] - self.x_end)
        opti.minimize(obj)

        # 初始条件约束
        opti.subject_to(X[:, 0] == self.x_start)
        opti.subject_to(X[:, -1] == self.x_end)

        # 动力学约束（Hermite-Simpson）
        for i in range(self.Nt):
            # 获取当前和下一个控制输入
            uk = U[:, i]
            ukp1 = U[:, i + 1]
            # 获取当前和下一个状态
            xk = X[:, i]
            xkp1 = X[:, i + 1]
            # 添加 Hermite-Simpson 约束
            res = dynamic_cons(self.params, xk, xkp1, uk, ukp1, self.dt)
            opti.subject_to(res == 0)
            # 可选：打印符号表达式以调试（建议仅在调试阶段使用）
            # print(f"Time step {i}: Residual = {res}")

        # 控制输入边界约束
        opti.subject_to(opti.bounded(self.lb_u, U, self.ub_u))

        # 设置初始猜测
        opti.set_initial(X, x_initial)
        opti.set_initial(U, u_initial)

        # 优化器选项
        p_opts = {
            "expand": False,
            "ipopt": {
                "print_level": 5,  # 增加日志详细程度
                "constr_viol_tol": 1e-4,  # 约束违反容忍度
                "acceptable_constr_viol_tol": 1e-4,  # 可接受的约束违反容忍度
                "tol": 1e-4,  # 整体收敛容差
                "dual_inf_tol": 1e-4,  # 对偶变量容忍度
                "compl_inf_tol": 1e-4  # 互补性容忍度
            }
        }
        s_opts = {
            "max_iter": 10000  # 增加最大迭代次数
        }
        opti.solver('ipopt', p_opts, s_opts)

        try:
            solution = opti.solve()
            x_opt = solution.value(X)
            u_opt = solution.value(U)
            return x_opt, u_opt
        except RuntimeError as e:
            print("优化求解失败:", e)
            return None, None


# 系统参数
params = {
    'mc': 1.0,  # 小车质量
    'mp': 0.2,  # 摆杆质量
    'l': 0.5  # 摆杆长度
}

# 状态和控制维度
Nx = 4
Nu = 1

# 权重矩阵
Q = ca.DM.eye(4) * 1
R = ca.DM.eye(1) * 0.1
Qf = ca.DM.eye(4) * 10

# 控制输入边界
lower_upper_bound_ux = {'lb_u': -10, 'ub_u': 10}

# 初始状态和目标状态
x_start = ca.MX.zeros(Nx)  # 修正维度，使用 MX
x_end = ca.vertcat(
    0,  # p
    ca.pi,  # theta (倒立位置)
    0,  # p_dot
    0  # theta_dot
)

# 时间设置
dt = 0.05
tf = 2.0
Nt = int(tf / dt)
t_vec = np.linspace(0, tf, Nt + 1)

# 初始化优化器
solver = DIRCOL(
    params=params,
    Q=Q,
    R=R,
    Qf=Qf,
    lower_upper_bound_ux=lower_upper_bound_ux,
    x_start=x_start,
    x_end=x_end,
    Nx=Nx,
    Nu=Nu,
    Nt=Nt,
    dt=dt
)
x_initial=0.001*np.random.rand(Nx,Nt+1)
u_initial=0.001*np.random.rand(Nu,Nt+1)
xks,uks=solver.solve(x_initial=x_initial,u_initial=u_initial)

# 绘制结果
if xks is not None and uks is not None:
    plt.figure(figsize=(12, 6))
    labels = ["p", "theta", "p_dot", "theta_dot"]
    for i in range(Nx):
        plt.plot(t_vec, xks[i, :], label=labels[i])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("State Trajectory")
    plt.grid(True)
    plt.savefig("state.png")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(t_vec, uks, label="u")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    plt.title("Control Trajectory")
    plt.grid(True)
    plt.savefig("control.png")
    plt.show()
else:
    print("优化求解失败，无法绘制轨迹。")

from matplotlib.animation import FuncAnimation
# 动画仿真参数
l = 0.5  # 摆杆长度
car_width = 0.2  # 小车宽度
car_height = 0.1  # 小车高度

def animate_inverted_pendulum(t_vec, xks, l):
    """
    基于优化结果进行动画仿真。

    参数：
    - t_vec: 时间序列
    - xks: 状态轨迹 [p, theta, p_dot, theta_dot]
    - l: 摆杆长度
    """
    # 提取小车位置和摆杆角度
    p_vals = xks[0, :]
    theta_vals = xks[1, :]

    # 初始化图形
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)

    # 绘制小车和摆杆
    car = plt.Rectangle((-car_width / 2, -car_height / 2), car_width, car_height, color="blue")
    pendulum_line, = ax.plot([], [], lw=2, color="darkorange")
    ax.add_patch(car)

    def init():
        """ 初始化动画元素 """
        car.set_xy((-car_width / 2, -car_height / 2))
        pendulum_line.set_data([], [])
        return car, pendulum_line

    def update(frame):
        """ 更新动画元素 """
        p = p_vals[frame]
        theta = theta_vals[frame]

        # 更新小车位置
        car.set_xy((p - car_width / 2, -car_height / 2))

        # 更新摆杆位置
        pendulum_x = [p, p + l * np.sin(theta)]
        pendulum_y = [0, -l * np.cos(theta)]
        pendulum_line.set_data(pendulum_x, pendulum_y)

        return car, pendulum_line

    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(t_vec), init_func=init, blit=True, interval=dt * 1000)
    anim.save("cart-pole-swingup.gif", writer="pillow", fps=30)
    print("the GIF graph of whole process has been saved successfully!")


# 调用动画函数
if xks is not None:
    animate_inverted_pendulum(t_vec, xks, l)

else:
    print("没有优化结果，无法生成动画。")