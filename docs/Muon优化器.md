# Muon 优化器

*本文撰写于 2025 年 12 月 19 日，最后更新于 2025 年 12 月 19 日*
*原博客链接：<https://kellerjordan.github.io/posts/muon/>*

## 算法概览
Muon 是专门针对**神经网络二维参数**的优化器，其算法流程可被概述如下：

* **Algorithm Muon**
* Require：Learning rate $$\eta$$, momentum $$\mu$$
* Initialize $$B_0 \leftarrow 0$$
* **for t = 1, 2, …, T do**
* $$\quad$$ Compute gradient $$G_t \leftarrow \nabla_\theta \mathcal L_t \left(\theta_{t-1}\right)$$
* $$\quad$$ Compute momentum $$B_t \leftarrow \mu B_{t-1} + G_t$$
* $$\quad$$ Orthogonalize $$O_t \leftarrow \text{NewtonSchulz}(B_t)$$
* $$\quad$$ Update parameters $$\theta_t \leftarrow \theta_{t-1} - \eta O_t$$ 
* **end for**
* return $$\theta_T$$

## 原理简介