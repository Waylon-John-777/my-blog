# 扩散模型系列（一）：从 Fokker-Plank 方程谈起

*本文撰写于 2026 年 01 月 19 日，最后更新于 2026 年 01 月 20 日*

## 为什么会有这篇文章

约莫在一年前我初学扩散模型的时候

## Fokker-Plank 方程推导

分布函数随时间的变化包括两部分的贡献：

1. 相空间中的输运；
2. 粒子相互作用所引起的碰撞；

$$\dfrac{\partial f}{\partial t} + \mathbf v \cdot \nabla f + \dfrac{\mathbf F}{m} \cdot \nabla_{\mathbf v} f = \left(\dfrac{\partial f}{\partial t}\right)_c \tag 1$$

假设碰前速度 $\mathbf v - \Delta \mathbf v$，碰后速度 $\mathbf v$，记 $\psi \left(\mathbf v - \Delta \mathbf v, \Delta \mathbf v\right)$ 为该过程发生的概率密度（e.g. $\int \psi \left(\mathbf v - \Delta \mathbf v, \Delta \mathbf v\right) \text d \Delta \mathbf v= 1$），显然我们有：

$$f(\mathbf r, \mathbf v, t) = \int  f(\mathbf r, \mathbf v - \Delta \mathbf v, t - \Delta t) \, \psi \left(\mathbf v - \Delta \mathbf v, \Delta \mathbf v\right) \text d \Delta \mathbf v \tag 2$$

不妨假设 $\|\Delta \mathbf v\| < \|\mathbf v\|$，对积分号内部在 $\mathbf v$ 处作二阶展开有：

$$
\begin{align}
\left[f\psi\right] \left(\mathbf r, \mathbf v - \Delta \mathbf v, \Delta \mathbf v, t - \Delta t\right) =  \; &\left[f\psi\right] \left(\mathbf r, \mathbf v, \Delta \mathbf v, t - \Delta t\right) - \Delta \mathbf v \cdot \dfrac{\partial}{\partial \mathbf v} \left[f\psi\right] \left(\mathbf r, \mathbf v, \Delta \mathbf v, t - \Delta t\right) \\ &+ \dfrac{1}{2}  \Delta \mathbf v \Delta \mathbf v : \dfrac{\partial^2}{\partial \mathbf v \partial \mathbf v} \left[f\psi\right] \left(\mathbf r, \mathbf v, \Delta \mathbf v, t - \Delta t\right)\tag 3
\end{align}
$$

上式两侧对 $\Delta \mathbf v$ 作积分，并取极限 $\displaystyle \lim_{\Delta t \rightarrow 0}$ 有：

$$\left(\dfrac{\partial f}{\partial t}\right)_c = - \dfrac{\partial}{\partial \mathbf v} \cdot \left[\mathbf A(\mathbf v) \, f(\mathbf r, \mathbf v, t)\right]  + \dfrac{1}{2} \dfrac{\partial^2}{\partial \mathbf v \partial \mathbf v} : \left[\mathbf B(\mathbf v) \, f(\mathbf r, \mathbf v, t)\right] \tag 4$$

其中 $\mathbf A(\mathbf v) := \displaystyle \lim_{\Delta t \rightarrow 0} \dfrac{1}{\Delta t} \displaystyle \int \psi(\mathbf v, \Delta \mathbf v) \Delta \mathbf v \, \text d \Delta \mathbf v $，$\mathbf B(\mathbf v) := \displaystyle \lim_{\Delta t \rightarrow 0} \dfrac{1}{\Delta t} \displaystyle \int \psi(\mathbf v, \Delta \mathbf v) \Delta \mathbf v \Delta \mathbf v \, \text d \Delta \mathbf v $.

## $\mathbf A(\mathbf v)$ 与 $\mathbf B(\mathbf v)$ 的物理意义