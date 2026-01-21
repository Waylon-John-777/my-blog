# 扩散模型系列（一）：从 Fokker-Plank 方程谈起

*本文撰写于 2026 年 01 月 19 日，最后更新于 2026 年 01 月 20 日*

## 为什么会有这篇文章

大约一年前初学扩散模型时，我对逆扩散过程感到困惑：<mark>这里的"逆"到底指的是什么</mark>？若按经典力学的意义去理解，所谓的去噪 [[1]](https://arxiv.org/abs/2006.11239) 不就是还原到原始图像，而非生成新的样本？DDPM 的原始论文通过构造变分下界进行优化，但这一角度难以触及问题的本质.

一个自然的思路是从统计物理的视角出发，例如随机微分方程 [[2]](https://arxiv.org/abs/2011.13456). 前向过程的构建较为直接，但反向过程的推导在 [[2]](https://arxiv.org/abs/2011.13456) 中着墨不多，仅引用了一篇早期文献. 沿此路径深入理解需要补充伊藤积分的相关知识，这对非数学背景的读者而言存在一定门槛.

因此，我尝试用所学的专业知识重新梳理反向过程的推导. 在这一框架下，所用到的工具仅限于普通的向量微积分. 本系列将呈现：**扩散模型中的"逆"，指的并非其它，正是分布函数意义上的逆变化**.

<div align="center">
  <img src="./figs/扩散模型系列（一）：从 Fokker-Plank 方程谈起/DDPM animation.gif" width="400"><br>
  <b>Fig 1. 托卡马克单零点放电位形下的逆扩散过程（摘自本人毕业论文）</b>
</div>

## Fokker-Plank 方程推导

分布函数随时间的变化包括两部分的贡献 [[3]]：

1. 相空间中的输运；
2. 粒子相互作用所引起的碰撞；

$$\dfrac{\partial f}{\partial t} + \mathbf v \cdot \nabla f + \dfrac{\mathbf F}{m} \cdot \nabla_{\mathbf v} f = \left(\dfrac{\partial f}{\partial t}\right)_c \tag 1$$

假设碰前速度 $\mathbf v - \Delta \mathbf v$，碰后速度 $\mathbf v$，记 $\psi \left(\mathbf v - \Delta \mathbf v, \Delta \mathbf v\right)$ 为该过程发生的概率密度（e.g. $\displaystyle \int \psi \left(\mathbf v - \Delta \mathbf v, \Delta \mathbf v\right) \text d \Delta \mathbf v= 1$），显然我们有：

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

## A(v) 与 B(v) 的物理意义

假设有试探粒子在 $t = 0$ 时刻满足分布 $f(\mathbf v) = \delta (\mathbf v - \mathbf u_0)$，代入式 $(4)$ 两侧后对速度求一阶矩有：

$$\dfrac{\partial \mathbf u}{\partial t}\mid_{\mathbf u = \mathbf u_0, \, t = 0} \;= \mathbf A(\mathbf u_0) \tag 5$$

显然 $\mathbf A(\mathbf v) \propto - \dfrac{\mathbf v}{\tau}$，即 $\mathbf A(\mathbf v)$ 体现了粒子由于碰撞而产生的平均速度变化率.
## 参考资料

[[1] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[[2] Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

[[3] 中国科学技术大学：等离子体动理学课堂讲义]