# LLM 强化学习不稳定的原因：训推差异与策略滞后

*本文撰写于 2025 年 12 月 11 日，最后更新于 2025 年 12 月 15 日*

## 强化学习目标函数：sequence-level vs token-level

将自回归 LLM 参数化为策略 $\pi_\theta$，并用 $x$ 表示采样自数据集 $\mathcal D$ 的 prompt 输入. LLM 生成回复 $y$ 的概率可被记为：

$$\pi_\theta (y | x) = \prod_{t=1}^T \pi_\theta (y_t | x, y_{<t})$$

在 LLM 的强化学习中，语言的上下文特性决定了需为整条回复 $y$ 设置单一标量奖励 $R(x, y)$. 为使奖励最大化，我们需对如下函数做极值优化：

$$\mathcal J^{seq} (\theta) = \mathbb E_{x \sim \mathcal D, \, y \sim \pi_\theta(\cdot | x)} \left[ \, R(x, y) \, \right]$$

**由于 RL 中的轨迹采样往往不是在训练引擎中完成（如 Megatron）而是在推理引擎中实现（如 SGLang 和 vLLM）**，我们需对上式做出如下修正：

$$\mathcal J^{seq} (\theta) = \mathbb E_{x \sim \mathcal D, \, y \sim \mu_{\theta_{old}}(\cdot | x)} \left[ \, \frac{\pi_\theta (y | x)}{\mu_{\theta_{old}} (y|x)} R(x, y) \, \right] \tag 3$$

$\mu$ 在这里代表推理引擎中的策略，它与训练引擎中的策略 $\pi$ 存在数值差异因此需要做重要性采样修正偏差. **$\frac{\pi_\theta (y | x)}{\mu_{\theta_{old}} (y|x)}$ 涉及长序列连乘，直接优化会导致方差巨大，完全不可训练**. 因此在工程实现中（如各类主流策略梯度算法），我们常对如下 token-level 的代理目标进行优化：

$$\mathcal J^{token}(\theta) = \mathbb E_{x \sim \mathcal D, \, y \sim \mu_{\theta_{old}}} \left[ \sum_{t=1}^T \frac{\pi_\theta (y_t | x, y_{<t})}{\mu_{\theta_{old}} (y_t|x, y_{<t})} R(x, y)\right] \tag 4$$

在下一节中，我们将证明在适当条件下 $\nabla_\theta \mathcal J^{token}$ 是 $\nabla_\theta \mathcal J^{seq}$ 的一阶近似.

## 一阶近似条件与影响因素

### token-level 是 sequence-level 的一阶近似

让我们做一假设，token-level 下 $\pi_\theta$ 与 $\mu_{\theta_{old}}$ 近似相等，即：

$$\dfrac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{old}}(y_t|x, y_{<t})} = 1 + \delta_t \tag 5$$

其中 $\delta_t$ 为扰动，满足 $|\delta_t| \ll 1$. 在 sequence-level 中忽略二阶及以上的小量则有：

$$\dfrac{\pi_\theta (y | x)}{\mu_{\theta_{old}}(y|x)} = \prod_{t=1}^T \dfrac{\pi_\theta (y_t | x, y_{<t})}{\mu_{\theta_{old}}(y_t|x, y_{<t})} \approx 1 + \sum_{t=1}^T \delta_t \tag 6$$

将其代入$\mathcal J^{seq}(\theta)$，两侧求梯度得：

$$\boxed{\nabla_\theta \mathcal J^{seq} \approx \nabla_\theta \mathcal J^{token}} \tag 7$$

这解释了在如 PPO, GRPO 等主流算法中，$\mathcal J^{token}(\theta)$ 被选作代理目标的合理性.

### 训推差异与策略滞后

自然地，读者很容易抛出一个疑问：在什么情况下，$|\delta_t| \ll 1$ 的近似条件会被破坏？为解决这个疑问，我们可以做如下改写：

$$\dfrac{\pi_\theta (y_t | x, y_{<t})}{\mu_{\theta_{old}}(y_t | x, y_{<t})} = \dfrac{\pi_{\theta_{old}} (y_t | x, y_{<t})}{\mu_{\theta_{old}}(y_t | x, y_{<t})} \cdot \dfrac{\pi_\theta (y_t | x, y_{<t})}{\pi_{\theta_{old}}(y_t | x, y_{<t})} \tag 8$$

**上式右侧第一项代表训推引擎的数值精度差异，第二项则代表 rollout 策略与 target 策略之间的不一致性**：越靠后被消耗的 mini-batch，其对应的样本往往具有更严重的 staleness.

因此，为保障 token-level 目标的一阶近似有效性，原则上应从两个方向缩小 $\pi_\theta$ 与 $\mu_{\theta_{old}}$ 之间的差距：一是减小训练与推理引擎间的数值差异，二是将策略陈旧度控制在适度范围内.

## MoE 带来的挑战

MoE 模型中 token-level 的重要性采样可改写如下：

$$\dfrac{\pi_\theta (y_t | x, y_{<t}, e_t^\pi)}{\mu_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\mu_{old}})} = \dfrac{\pi_{\theta_{old}} (y_t | x, y_{<t}, e_t^{\pi_{old}})}{\mu_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\mu_{old}})} \cdot \dfrac{\pi_\theta (y_t | x, y_{<t}, e_t^\pi)}{\pi_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\pi_{old}})} \tag 9$$

其中 $e_t^\pi$ 与 $e_t^{\mu_{old}}$ 分别代表训练/推理引擎中，新/旧策略下模型处理第 $t$ 个 token 时的专家. 从上式不难发现，**路由机制的引入将与训推差异及策略滞后相互交织，进而导致一阶近似假设更易失效**.

我们可以通过路由重演机制缓和以上的问题，其核心思想是在策略优化过程中固定路由选定的专家，以此稳定强化学习训练过程. 具体而言，有以下两种实现思路：

一、重演 rollout 策略在训练引擎中确定的路由专家，该方法旨在减轻策略滞后的影响：

$$\dfrac{\pi_\theta (y_t | x, y_{<t}, \color{blue}{e_t^{\pi_{old}}})}{\mu_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\mu_{old}})} = \dfrac{\pi_{\theta_{old}} (y_t | x, y_{<t}, e_t^{\pi_{old}})}{\mu_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\mu_{old}})} \cdot \underbrace{\dfrac{\pi_\theta (y_t | x, y_{<t}, \color{blue}{e_t^{\pi_{old}}})}{\pi_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\pi_{old}})}}_{\text {policy staleness} \, \downarrow} \tag {10}$$

二、重演 rollout 策略在推理引擎中确定的路由专家，该方法对数值差异及策略滞后均有所缓和：

$$\dfrac{\pi_\theta (y_t | x, y_{<t}, \color{blue}{e_t^{\mu_{old}}})}{\mu_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\mu_{old}})} = \underbrace{\dfrac{\pi_{\theta_{old}} (y_t | x, y_{<t}, \color{blue}{e_t^{\mu_{old}}})}{\mu_{\theta_{old}}(y_t|x, y_{<t}, e_t^{\mu_{old}})}}_{\text {numerical differences} \, \downarrow} \cdot \underbrace{\dfrac{\pi_\theta (y_t | x, y_{<t}, \color{blue}{e_t^{\mu_{old}}})}{\pi_{\theta_{old}}(y_t|x, y_{<t}, \color{blue}{e_t^{\mu_{old}}})}}_{\text {policy staleness} \, \downarrow} \tag {11}$$

值得注意的是，尽管路由重演机制缓和了 MoE 中被放大的训推差异与策略滞后问题，**它不可避免的在优化目标中引入了额外的偏差：每个 token $y_t$ 所对应的专家 $e^\pi_t$ 本应由模型自主决定，而该机制对其进行了人为干预**.

## 参考文献
[1] **<https://arxiv.org/abs/2512.01374>**