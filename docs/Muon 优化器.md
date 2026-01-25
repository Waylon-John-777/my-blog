# Muon 优化器

*本文撰写于 2025 年 12 月 19 日，最后更新于 2025 年 12 月 22 日*

## 算法概览
Muon 是专门针对**神经网络二维参数**的优化器，其算法流程可被概述如下 [[1]](<https://kellerjordan.github.io/posts/muon/>)：

* **Algorithm Muon**
* Require：Learning rate $\eta$, momentum coefficient $\mu$
* Initialize $M_0 \leftarrow 0$
* **for t = 1, 2, …, T do**
* $\quad$ Compute gradient $G_t \leftarrow \nabla_\theta \mathcal L_t \left(\theta_{t-1}\right)$
* $\quad$ Compute momentum $M_t \leftarrow \mu M_{t-1} + G_t$
* $\quad$ Orthogonalize $O_t \leftarrow \text{NewtonSchulz}(M_t)$
* $\quad$ Update parameters $\theta_t \leftarrow \theta_{t-1} - \eta O_t$ 
* **end for**
* return $\theta_T$

其中 NewtonSchulz 迭代在 PyTorch 中的写法如下 [[1]](<https://kellerjordan.github.io/posts/muon/>)：

```python
def newtonschulz5(M, steps=5, eps=1e-7):
    assert M.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M.bfloat16()
    X /= (X.norm() + eps)
    if M.size(0) > M.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if M.size(0) > M.size(1):
        X = X.T
    return X
```
## 原理简介
### Muon 的本质

Muon 旨在对给定矩阵 $M$，寻找 F 范数意义下与其最接近的半正交矩阵：

$$\operatorname*{arg\,min}_{O} \|O - M\|_F, \quad \text{either} \, O^\top O = I \, \text{or} \, O O^\top = I \tag 1$$

数学上可证明满足如上条件的半正交阵 $O = UV^\top$，其中 $U, V$ 分别为 $M$ 的左右奇异向量矩阵.
### NewtonSchulz 迭代到底在做什么
NS 迭代实际上是在逼近待求半正交阵 $O$. 设 $X \in \mathbb R^{m \times n} \; (m \le n) \,$ 作奇异值分解有 $X = U \Sigma V^\top$，其中 $U \in \mathbb R^{m \times m},$ $\Sigma \in \mathbb R^{m \times m},$ $V \in \mathbb R^{n \times m}$，单步 NS 迭代给出如下结果：

$$X = aU \Sigma V^\top + bU\Sigma^3V^\top + cU\Sigma^5V^\top\tag 2$$

不妨设 $\Phi(\Sigma) = a \Sigma + b \Sigma^3 + c \Sigma^5$，k 步 NS 迭代后有：

$$X = U \, \Phi^k(\Sigma) \, V^\top \tag 3$$

**为避免发散，$X$ 的奇异值需落在区间 $[0, 1]$ 内**，这已经通过作归一化 $X = \dfrac{M}{\|M\|_F}$ 实现（注意到 $\|M\|_F = \sqrt{\sum {\sigma_i^M}^2}$）. 显然迭代的最终效果依赖于超参 $a, b, c$ 的选取，让我们首先看看 $\Phi^k$ 在 $a = 2, b = -1.5, c = 0.5$ 的情况：

<div align="center">
  <img src="./figs/Muon 优化器/muon1.png" width="400"><br>
  <b>Fig 1. 迭代数 k 增加时，NS 收敛于 1</b>
</div>

不难发现在该超参下（并不唯一），NewtonSchulz 迭代可实现近似正交化的高效求解. 

### 为何要近似正交化

很容易有这样的疑问，为什么要对动量做正交化？

这事实上来源于经验观察：SGD-momentum/Adam 对 Transformer 的 2D 参数的更新满足低秩特性，也即**更新几乎仅沿少数几个奇异方向**，这使得 $\sigma_i$ 较小的信号被淹没，网络在一个极度各向异性的子空间中振荡.

<mark>直觉上而言，通过抹平奇异值幅度的差异（$O = U V^\top$），Muon 使优化在高维参数空间中更加各向同性和稳定</mark>.

## 系数的确立 & 转置的作用
### 默认系数的由来

系数的选取遵循以下两点的考量 [[1]](<https://kellerjordan.github.io/posts/muon/>)：

1. $\Phi’(0) = a$ 决定了初始微小奇异值收敛的速度，所以应尽可能增大；

2. NS 迭代不必要求结果严格等于 1，只需落入区间 $[1 - \epsilon, 1 + \epsilon]$；

经验观察表明 $\epsilon$ 可高达 0.3. 下图绘制了迭代次数为 5 时 [[1]](<https://kellerjordan.github.io/posts/muon/>) 中默认参数设置下的收敛情况：

<div align="center">
  <img src="./figs/Muon 优化器/muon2.png" width="400"><br>
  <b>Fig 2. 默认参数设置下的 NS 迭代</b>
</div>

### 计算量分析

设有 $A \in \mathbb R^{m \times k}, B \in \mathbb R^{k \times n}$，则矩阵乘 $AB$ 的计算量约为 $2mnk$. 计算后不难得到对于 $\mathbb R^{m \times n}$ 的单步迭代，略去二阶量后的 FLOPs 大小为 $2m^3 + 4m^2n$（从中也能看出做矩阵转置的必要性）.

假设神经网络参数对应的是一个线性层 $W \in \mathbb R^{n \times m}$，输入张量的形状为 $X \in \mathbb R^{b \times n}$，则一次正常训练步骤（前向与反向）的基准 FLOPs 数为 $6bmn$. 故在迭代次数为 $T$ 的设置下，Muon 的额外开销为 $\dfrac{mT}{b}$. 在 NanoGPT speedrunning 和 Llama 405B 训练的场景下，代入具体数据可得 Muon 的额外 FLOPs 占比分别为 0.7% 与 0.5% [[1]](<https://kellerjordan.github.io/posts/muon/>).

## 模型扩展后的表现：KIMI 的改进

对于形如 $[A, B]$ 的满秩矩阵，易证 MUON 更新量的 RMS 值为 $\sqrt{\dfrac{1}{\max (A, B)}}$，这带来了两个问题 [[2]](<https://arxiv.org/pdf/2502.16982>)：

1. 当 $\max (A, B)$ 较大，更新量过小，从而限制了模型的表示能力并导致次优性能；
2. 当 $\max (A, B)$ 较小，更新量过大，这会导致训练的不稳定性；

<mark>AdamW 常用以与 Muon 结合以更新如 RMSNorm, LM head 和 embedding 等非矩阵参数</mark>，为使超参数如学习率 $\eta$，权重衰减 $\lambda$ 在矩阵和非矩阵参数之间共享，KIMI 采用如下调整**将 MUON 的更新 RMS 匹配为与 AdamW 相似的 RMS**（经验观察 AdamW 的更新均方根值通常在 0.2 到 0.4 之间）：

$$\mathbf W_t = \mathbf W_{t - 1} - \eta_t \left(0.2 \cdot \sqrt{\max(A, B)} \cdot \mathbf O_t + \lambda \mathbf W_{t - 1}\right) \tag 4$$

## 参考资料

[[1] Muon: An optimizer for hidden layers in neural networks](<https://kellerjordan.github.io/posts/muon/>)

[[2] MUON IS SCALABLE FOR LLM TRAINING](<https://arxiv.org/pdf/2502.16982>)

---

## 引用本文

```bibtex
@misc{waylonblog2025,
  author = {Waylon John},
  title = {Muon 优化器},
  year = {2025},
  url = {https://waylon-john-777.github.io/my-blog/#/Muon%20优化器}
}
```