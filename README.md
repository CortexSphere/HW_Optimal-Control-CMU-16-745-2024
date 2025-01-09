分析一个平面双旋翼系统，使该双旋翼系统到达特定的状态（如特定的位置）。

$$
\ddot{x} = -(u_1 + u_2) \sin(\theta)
$$

$$
\ddot{y} = (u_1 + u_2) \cos(\theta) - mg
$$

$$
J\ddot{\theta} = \frac{1}{2}l(u_2 - u_1)
$$


这是一个非线性系统（**系统动力学**存在 $u$ 与 $x$ 的耦合）。将飞机在悬停状态线性化，此时有条件：

$\theta = 0, \quad u_1 = u_2 = \frac{1}{2}mg,$

并假设飞机一直在小角度附近运动，即 $\theta \approx 0$。在悬停状态线性化并不意味着飞机会一直处于这种状态，只是说我们假设飞机一直在悬停态附近运动。飞机仍然可以有 $\Delta u$、$\Delta \theta$。此时线性化的模型是较为标准的，对线性化的模型离散化后进行 LQR 和 MPC 控制。

---

由于此时问题不再是回到原点，而是到达任意一个状态 $x_{ref}$，因此H步的总成本函数J变为：

$$
\min_{x_{1:H+1}, u_{1:H-1}} \sum_{k=1}^{H-1} \left[ \frac{1}{2}(x_k - x_{ref})^T Q_k (x_k - x_{ref}) + \frac{1}{2}u_k^T R_k u_k \right] + (x_H - x_{ref})^T Q_H(x_H - x_{ref})
$$

进一步展开后为：

$$
\min_{x_{1:H+1}, u_{1:H-1}} \sum_{k=1}^{H-1} \left[ \frac{1}{2} x_k^T Q_k x_k + \frac{1}{2} u_k^T R_k u_k - (Q_k x_{ref})^T x_k \right] + x_H^T P x_H - (Q_H x_{ref})^T x_H
$$

约束条件为：

$$
x_{min} \leq x \leq x_{max}, \quad u_{min} \leq u \leq u_{max}
$$


其中：

- $Q_{k}$, $R_{k}$, $Q_{H}$ 分别为状态代价、输入代价和最终状态代价的权重矩阵；
- $x_{k}$ 表示系统的状态；
- $u_{k}$表示系统的控制输入；
- $x_{ref}$ 是目标状态；
- $H$ 是优化时间窗口的长度。


依旧可以转化成标准的 QP 问题（这里 $Q_k = Q$, $R_k = R$。像上一节所做，我们把决策变量堆在一起成 $z$）。

在求解问题时，用的是 OSQP，取 $N = 20$，因为 $dim(x) = 6$，$dim(u) = 2$（实际上 $x_2$ 到 $x_{21}$ 是决策变量，$u$ 由 $u_1$ 到 $u_{20}$ 是决策变量），所以优化变量一共是 160 维。

由于 OSQP 只存在不等式约束 $lower \leq Dz \leq upper$，因此所有等式约束都要转化成一个不等式约束，其中 $lower = upper$。

---

变量堆叠如下：

$$
z = 
\begin{bmatrix}
u_1 \\
u_2 \\
x_2 \\
\vdots \\
x_{20} \\
u_{20} \\
x_{21}
\end{bmatrix} \in \mathbb{R}^{160}
$$

---

矩阵形式如下：

$$
H =
\begin{bmatrix}
R & 0 & \cdots & 0 & 0 \\
0 & Q & \cdots & 0 & 0 \\
0 & 0 & Q & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & 0 \\
0 & 0 & 0 & 0 & P
\end{bmatrix}
\in \mathbb{R}^{160 \times 160}
$$


$$
D =
\begin{bmatrix}
B_1 & (-I) & \cdots & \cdots & 0 \\
0 & A & B & (-I) & \cdots & 0 \\
0 & 0 & \cdots & A_{20} & B_{20} & (-I) \\
- & - & - & - & - & - \\
S_u & & & & & \\
- & - & - & - & - & - \\
S_\theta & & & & & \\
\end{bmatrix}
\in \mathbb{R}^{(120+40+20) \times 160}
$$


---

上下限如下：

$$
lower =
\begin{bmatrix}
-A_1 x_1 \\
0 \\
\vdots \\
0 \\
u_{min} \\
\vdots \\
\theta_{min}
\end{bmatrix}
\in \mathbb{R}^{(120+40+20)},
\quad
upper =
\begin{bmatrix}
-A_1 x_1 \\
0 \\
\vdots \\
0 \\
u_{max} \\
\vdots \\
\theta_{max}
\end{bmatrix}
\in \mathbb{R}^{(120+40+20)}
$$

---

其中，约束共有 120 个状态的等式约束（每条 $x_{k+1} = A_k x_k + B u_k$ 可提供 $dim(x)$ 个约束），40 个关于输入 $u$ 的约束（$u$ 一共有 40 个，其中 $S_u \in \mathbb{R}^{40 \times 160}$ 是选择矩阵，使得 $S_u z = u$），20 个关于状态 $\theta$ 的约束（其中 $S_\theta \in \mathbb{R}^{20 \times 160}$ 是选择矩阵，使得 $S_\theta z = \theta$）。
