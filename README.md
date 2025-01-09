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


