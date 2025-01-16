import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.animation import FuncAnimation
from DIRCOL import trajectory_generate
from scipy.signal import cont2discrete
def dynamic_function(params,x,u):
    mc, mp, l = params["mc"], params["mp"], params["l"]
    g = 9.81
    p, theta, p_dot, theta_dot = x[0], x[1], x[2], x[3]
    s=np.sin(theta)
    c=np.cos(theta)
    H=np.array([[mc+mp, mp*l*c],
                [mp*l*c, mp*l**2]])
    C=np.array([[0,-mp*theta_dot*l*s],
                [0,0]])
    G=np.array([[0,mp*g*l*s]]).T
    B=np.array([[1,0]]).T
    qd=np.vstack((p_dot,theta_dot))
    qdd = np.linalg.solve(-H, C @ qd + G - B * u).flatten()
    return np.array([p_dot, theta_dot, qdd[0], qdd[1]])


def linearize_dynamics(params, x, u,eps=1e-3):
    A=np.zeros((4,4))
    for i in range(4):
        x_new=np.copy(x)
        x_new[i]+=eps
        A[:,i]=(dynamic_function(params,x_new,u)-dynamic_function(params,x,u))/eps
    u_new=np.copy(u)
    u_new+=eps
    B=(dynamic_function(params,x,u_new)-dynamic_function(params,x,u))/eps
    B = B.reshape(-1, 1)
    return A,B




class TV_LQR:
    def __init__(self, x_start, x_goal, x_ref,u_ref,Q,R,Qf,Nt,dt,params):
        self.params = params
        self.x_start = x_start
        self.x_goal = x_goal
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.dt = dt
        self.Nt = Nt
    def rk4_dynamics(self, x, u):
        f1=dynamic_function(self.params,x,u)
        f2=dynamic_function(self.params,x+0.5*self.dt*f1,u)
        f3=dynamic_function(self.params,x+0.5*self.dt*f2,u)
        f4=dynamic_function(self.params,x+self.dt*f3,u)
        return x+(self.dt/6.0)*(f1+2*f2+2*f3+f4)

    def discretize_system_foh(self,A_cont, B_cont):
        """
        将连续时间系统矩阵 A 和 B 离散化，使用一阶保持（FOH）。

        参数：
        - A_cont: 连续时间系统的 A 矩阵
        - B_cont: 连续时间系统的 B 矩阵
        - dt: 采样时间步长

        返回：
        - Ad: 离散时间系统的 A 矩阵
        - Bd: 离散时间系统的 B 矩阵
        """
        Ad, Bd = cont2discrete(
            (A_cont, B_cont, np.eye(A_cont.shape[0]), np.zeros((A_cont.shape[0], B_cont.shape[1]))), self.dt, method='foh')[
                          :2]
        return Ad, Bd

    def tv_lqr(self):
        """
        计算时间变化的LQR反馈增益列表
        """
        K_list = [np.zeros((1, 4)) for _ in range(self.Nt - 1)]
        P_list = [np.zeros((4, 4)) for _ in range(self.Nt)]
        P_list[-1]=self.Qf
        # 逆向迭代
        for i in reversed(range(self.Nt - 1)):
            P=P_list[i+1]
            x = self.x_ref[:, i]
            u = self.u_ref[i]
            A_cont, B_cont = linearize_dynamics(self.params, x, u, eps=self.dt)
            # 离散化 A 和 B
            A,B = self.discretize_system_foh(A_cont, B_cont)
            # 计算K
            K=np.linalg.solve(self.R+B.T@P@B,B.T@P@A)
            K_list[i] = K
            # 更新P
            P_list[i]= A.T @ P @ A - A.T @ P @ B @ K + self.Q
        return K_list

    def simulate(self):
        """
        使用计算得到的反馈增益列表模拟系统轨迹
        """
        K_list = self.tv_lqr()
        x_sim = np.zeros((4, self.Nt))  # 正确初始化为 (4, Nt)
        u_sim = np.zeros(self.Nt - 1)   # 初始化为 (Nt-1, )

        x_sim[:, 0] = self.x_start
        for i in range(self.Nt - 1):
            # 计算控制输入
            delta_x = x_sim[:, i] - self.x_ref[:, i]
            u = self.u_ref[i] - K_list[i] @ delta_x
            #u = np.clip(u, -10, 10)
            u_sim[i] = u
            # 积分得到下一个状态
            x_sim[:, i + 1] = self.rk4_dynamics(x_sim[:, i], u)
        return x_sim, u_sim

def plot_static(time, x_ref, x_real, u_real):
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    state_labels = ['Position p (m)', 'Angle θ (rad)', 'Velocity p_dot (m/s)', 'Angular Velocity θ_dot (rad/s)']
    for i in range(4):
        axs[i].plot(time, x_ref[i, :], label='Reference')
        axs[i].plot(time, x_real[i, :], label='Actual')
        axs[i].set_ylabel(state_labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[4].plot(time[:-1], u_real, label='Control Input u')
    axs[4].set_ylabel('Control Input (N)')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.savefig('state_control_static_plots.png')
    plt.show()

def plot_animation(time, x_ref, x_real, params):
    p_ref = x_ref[0, :]
    theta_ref = x_ref[1, :]
    p_real = x_real[0, :]
    theta_real = x_real[1, :]

    l = params['l']
    # 计算参考和实际的摆杆端点位置
    ref_x = p_ref
    ref_y = np.zeros_like(p_ref)  # 假设小车在y=0
    ref_pendulum_x = p_ref + l * np.sin(theta_ref)
    ref_pendulum_y = -l * np.cos(theta_ref)

    real_x = p_real
    real_y = np.zeros_like(p_real)
    real_pendulum_x = p_real + l * np.sin(theta_real)
    real_pendulum_y = -l * np.cos(theta_real)

    # 设置绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim((min(np.min(ref_pendulum_x), np.min(real_pendulum_x)) - l,
                 max(np.max(ref_pendulum_x), np.max(real_pendulum_x)) + l))
    ax.set_ylim((-l * 1.5, l * 1.5))
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Reference vs Actual Trajectory Tracking')

    # 绘制参考轨迹
    ref_cart = patches.Rectangle((ref_x[0] - 0.2, ref_y[0] - 0.1), 0.4, 0.2, linewidth=1, edgecolor='blue', facecolor='lightblue', label='Reference Cart')
    ref_pendulum, = ax.plot([], [], 'o-', color='blue', lw=2, label='Reference Pendulum')

    # 绘制实际轨迹
    real_cart = patches.Rectangle((real_x[0] - 0.2, real_y[0] - 0.1), 0.4, 0.2, linewidth=1, edgecolor='red', facecolor='lightcoral', label='Actual Cart')
    real_pendulum, = ax.plot([], [], 'o-', color='red', lw=2, label='Actual Pendulum')

    ax.add_patch(ref_cart)
    ax.add_patch(real_cart)
    ax.legend()

    def init():
        ref_pendulum.set_data([], [])
        real_pendulum.set_data([], [])
        ref_cart.set_xy((ref_x[0] - 0.2, ref_y[0] - 0.1))
        real_cart.set_xy((real_x[0] - 0.2, real_y[0] - 0.1))
        return ref_pendulum, real_pendulum, ref_cart, real_cart

    def update(frame):
        # 更新参考摆杆
        ref_pendulum.set_data([ref_x[frame], ref_pendulum_x[frame]],
                              [ref_y[frame], ref_pendulum_y[frame]])
        # 更新实际摆杆
        real_pendulum.set_data([real_x[frame], real_pendulum_x[frame]],
                               [real_y[frame], real_pendulum_y[frame]])
        # 更新小车位置
        ref_cart.set_xy((ref_x[frame] - 0.2, ref_y[frame] - 0.1))
        real_cart.set_xy((real_x[frame] - 0.2, real_y[frame] - 0.1))
        return ref_pendulum, real_pendulum, ref_cart, real_cart

    ani = FuncAnimation(fig, update, frames=range(0, len(time), max(1, len(time)//300)),
                        init_func=init, blit=True, repeat=False)

    ani.save("trajectory_tracking_animation.gif", writer="pillow", fps=30)
    print("the GIF graph of whole process has been saved successfully!")
if __name__=='__main__':
    #get the ideal trajectory
    x_ref, u_ref, time = trajectory_generate()
    # 状态和控制维度
    Nx = 4
    Nu = 1
    dt = 0.05
    tf = 2.0
    #true params different from that in DIRCOL
    params = {
        'mc': 1.05,
        'mp': 0.21,
        'l': 0.48
    }
    x_start = np.array([0,0,0,0]).T
    x_goal = np.array([0,np.pi,0,0]).T
    # 权重矩阵
    Q = np.diag([100,1,0.05,0.1])
    R = 0.01
    Qf = Q*100
    Nt = int(tf / dt)
    solver=TV_LQR(x_start, x_goal, x_ref,u_ref,Q,R,Qf,Nt,dt,params)
    x_real,u_real=solver.simulate()
    # 绘制静态图
    plot_static(time, x_ref, x_real, u_real)

    # 绘制并保存动画
    plot_animation(time, x_ref, x_real, params)












