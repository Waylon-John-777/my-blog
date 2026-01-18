# Multi-Head Latent Attention
## 多头注意力简要回顾

设 $d$ 为嵌入维度，$n_h$ 为注意力头的数量，$d_h$ 为每个注意力头的维度，并且 $\mathbf h_t \in \mathbb R^d$ 表示在某一注意力层中第 $t$ 个 token 的注意力输入向量. 标准的多头注意力计算首先通过投影矩阵 $W^Q, W^K, W^V \in \mathbb R^{n_hd_h \times d}$ 生成 $\mathbf q_t, \mathbf k_t, \mathbf v_t \in \mathbb R^{n_hd_h}$ 向量：

$$
\begin{align}
\mathbf q_t &= W^Q \, \mathbf h_t \tag 1 \\
\mathbf k_t &= W^K \, \mathbf h_t \tag 2 \\
\mathbf v_t &= W^V \, \mathbf h_t \tag 3
\end{align}
$$

$\mathbf q_t, \mathbf k_t, \mathbf v_t$ 随即将被切分为 $n_h$ 份：

$$
\begin{align}
\mathbf q_t = [\mathbf q_{t,1}; \mathbf q_{t, 2}; \dots ; \mathbf q_{t, n_h}] \tag 4 \\
\mathbf k_t = [\mathbf k_{t,1}; \mathbf k_{t, 2}; \dots ; \mathbf k_{t, n_h}] \tag 5 \\
\mathbf v_t = [\mathbf v_{t,1}; \mathbf v_{t, 2}; \dots ; \mathbf v_{t, n_h}] \tag 6 \\
\end{align}
$$

以进行随后的多头计算：

$$
\begin{align}
\mathbf o_{t, i} = \sum_{j=1}^t \text{softmax}_j \left(\dfrac{\mathbf q_{t, i}^\top \mathbf k_{j, i}}{\sqrt {d_h}}\right) \mathbf v_{j, i} \tag 7 \\
\mathbf u_t = W^O [\mathbf o_{t,1}; \mathbf o_{t, 2}; \dots ; \mathbf o_{t, n_h}] \tag 8
\end{align} 
$$

其中 $\mathbf q_{t, i}, \mathbf k_{t, i}, \mathbf v_{t, i} \in \mathbb R^{d_h}$ 代表第 $i$ 个注意力头的 query, key 和 value 向量. $W^O \in \mathbb R^{d \times n_hd_h}$ 代表投影矩阵. **传统的 MHA 对于每个 token 的 KV Cache 大小为 $2n_hd_hl$，其中 $l$ 为模型层数**.
## MLA：将 KV 压缩为潜向量

让我们重新审视一下如上的计算过程，假设 $W^K$ 与 $W^V$ 可以做如下近似分解：

$$
\begin{align}
W^K \approx W_U^K \, W_D^{K, V} \tag 9 \\
W^V \approx W_U^V \, W_D^{K, V} \tag {10}
\end{align} 
$$

其中 $W_U^K, W_U^V \in \mathbb R^{n_hd_h \times d_c}$，$W_D^{K, V}, \in \mathbb R^{d_c \times d}$ 分别为上投影与下投影矩阵. 不难得到：

$$
\begin{align}
\mathbf c_t^{K, V} =  W_D^{K, V} \mathbf h_t \tag {11} \\
\mathbf k_t^{C} = W_U^K  \mathbf c_t^{K, V} \tag {12} \\
\mathbf v_t^{C} = W_U^V  \mathbf c_t^{K, V} \tag {13}
\end{align} 
$$

$\mathbf c_t^{K, V} \in \mathbb R^{d_c}$ 为 key 和 value 的压缩潜在表示，$d_c \, (\ll n_hd_h)$ 为压缩维度. 推理阶段，MLA 仅需缓存 $\mathbf c_t^{K, V}$，故单一 token 的 KV Cache 大小为 $d_cl$，**压缩比为 $\dfrac{d_c}{2n_hd_h}$.**

## 上投影矩阵的合并

我们在 MLA 的设置下重新审视一下式 $(7)$ 与式 $(8)$：

$$
\begin{align}
\mathbf q_{t, i}^\top \mathbf k_{j, i} &= \mathbf h_t^\top {W_i^Q}^\top W_i^K \mathbf h_j \\
&= \mathbf h_t^\top \underbrace{{W_i^Q}^\top W_{U, i}^K}_{{\tilde W_i^Q}^\top}  W_D^{K, V} \mathbf h_j \\ 
&= \mathbf h_t^\top {\tilde W_i^Q}^\top \underbrace{W_D^{K, V} \mathbf h_j}_{\mathbf c_j^{K, V}} \tag {14}
\end{align} 
$$

$$
\begin{align}
\mathbf u_t &= \sum_i W_i^O \mathbf o_{t, i} \\ 
&= \sum_i W_i^O \sum_{j=1}^t \underbrace{\text{softmax}_j \left(\dfrac{\mathbf q_{t, i}^\top \mathbf k_{j, i}}{\sqrt {d_h}}\right)}_{\alpha_{t, j}^{(i)}} \mathbf v_{j, i} \\ 
&= \sum_i \underbrace{W_i^O W_{U, i}^V}_{\tilde W_i^O}\sum_{j=1}^t \alpha_{t, j}^{(i)} \underbrace{W_D^{K, V} \mathbf h_j}_{\mathbf c_j^{K, V}} \tag {15}
\end{align} 
$$

可见上投影阵 $W_U^K$ 和 $W_U^V$ 在实际计算中可以被以某种形式融入新矩阵 $\tilde W^Q$ 和 $\tilde W^O$ 的构建.
## RoPE 解耦

## 矩阵计算与代码展示

---

## 引用本文

```bibtex
@misc{waylonblog2026,
  author = {Waylon John},
  title = {Multi-Head Latent Attention},
  year = {2026},
  url = {https://waylon-john-777.github.io/my-blog/#/MLA}
}
```