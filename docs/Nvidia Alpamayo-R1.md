# Nvidia Alpamayo-R1

*本文撰写于 2026 年 03 月 06 日，最后更新于 2026 年 03 月 06 日*

## 总体概览

构建面向自动驾驶的 VLA 需要赋予其一系列超越当前 VLM 现有能力的新特性，包括：

1. 多相机、多时间步观测的高效编码；
2. 因果结构化推理；
3. 实时、精确且满足运动学约束的多峰轨迹预测；
4. 推理-动作对齐；

为应对上述挑战，Nvidia 的研究人员提出了 Alpamayo-R1 (AR1)——一种模块化的 VLA 架构. 它的设计哲学包括：

* 灵活性：可采用任意现成 VLM 骨干网络；
* 模块化：引入领域特定组件以实现高效视觉编码和实时动作解码；

与仅预测轨迹的基线相比，AR1 在困难场景下的规划精度提升最高达 12%，在闭环仿真中近距离冲突率降低 35%. RL 后训练使推理质量提升 45%，推理-动作一致性提升 37%. 模型从 0.5B 扩展到 7B 参数时性能持续提升. 实车路测验证了实时性能（端到端延迟 99 ms）及城市道路部署的可行性. 通过桥接可解释推理与精确控制，AR1 展示了一条通向 L4 级自动驾驶的可行路径.
  
## 问题建模

给定历史观测序列 $o$，包括多相机图像 $o_{\text{image}}$ 和自车运动历史 $o_{\text{egomotion}}$，AR1 被训练执行两项任务：生成推理，并预测自车未来轨迹 $\tau = \{(x^i, y^i, \theta^i)\}_{i=1}^{64}$. 

<div align="center">
  <img src="./figs/Nvidia Alpamayo-R1/Backbone.png" width="400"><br>
  <b>Fig 1. Alpamayo-R1 架构一览</b>
</div>

其中 $(x^i, y^i, \theta^i)$ 表示第 $i$ 个时间采样点下车辆在 BEV 平面的位置坐标和偏航角，在 10 HZ 的频率下，该序列给出了未来 6.4 秒的状态预测.

值得注意的是，AR1 并不直接回归位置序列 $\tau$，而是采用基于 unicycle 运动学模型的中间表示. 模型实际预测的是 $u = \{(a^i, \kappa^i)\}_{i=1}^{64}$，其中 $a^i$ 为加速度，$\kappa^i$ 为曲率. 训练时，通过带 L2 正则的最小二乘拟合原始轨迹 $\tau^*$，反解出 ground-truth $u^*$；推理时再通过式 $(1)$ 的离散化运动方程将 $u$ 还原为轨迹 $\tau$. 论文指出，直接在 $x$-$y$ 空间回归路点容易受传感器噪声影响，而参数化后的表示天然具备运动学约束，闭环性能更优.

$$
\begin{pmatrix} 
x^{i+1} \\ y^{i+1} \\ \theta^{i+1} \\ v^{i+1}
\end{pmatrix}
 =                                                              
\begin{pmatrix}                                                
x^i + \dfrac{\Delta T}{2} \left( v^i \cos\theta^i + v^{i+1}    
\cos\theta^{i+1} \right) \\[6pt]
y^i + \dfrac{\Delta T}{2} \left( v^i \sin\theta^i + v^{i+1}
\sin\theta^{i+1} \right) \\[6pt]
\theta^i + \Delta T \, \kappa^i v^i + \dfrac{\Delta T^2}{2} \,
\kappa^i a^i \\[6pt]
v^i + \Delta T \, a^i
\end{pmatrix} \tag 1
$$

## 视觉编码

VLA 需要部署在车载设备上，因此视觉编码器必须在尽可能少的 token 数量下保留环境的语义信息. 论文主要讨论了三种视觉 token 化策略：

### 1. Single-Image Tokenization

AR1 的默认方案. 使用 ViT 将每张图像分为 patch 并编码为 1D token 序列 $\mathbf f \in \mathbb R^{\frac{H}{14} \times \frac{W}{14} \times D}$，再经 2× 双线性下采样转化为 $\mathbf f' \in \mathbb R^{\frac{H}{28} \times \frac{W}{28} \times D}$. 举例而言，448×280 像素的图像产生 160 个 token. 该方案实现简单，但 token 数量会随图像分辨率和摄像头数量线性增长. 由于自动驾驶车辆通常使用 6 到 10 个摄像头以获得 360 度视野，基于 patch 的 token 化将在每个时间步产生上千个 token，从而使实时推理变得不可行.

### 2. Multi-Camera Tokenization

为解决上述问题，AR1 因此支持使用基于 triplane 的多摄像头编码器：先将多个摄像头图像编码并投影到三个正交的 2D 特征平面上. 由于 triplane 尺寸固定，token 数量与摄像头数量和图像分辨率解耦. 在典型配置下（$S_x = S_y = 96, S_z = 48, p = 8$），每个时间步仅需 288 个 token，7 摄像头场景下等效每张图像约 41 个 token，相比单图像方案压缩约 3.9 倍，且端到端驾驶指标无明显下降.

### 3. Multi-Camera Video Tokenization

尽管 triplane 方案已大幅减少了表示传感器观测所需的 token 数量，但考虑到帧间存在信息冗余以及结构化特征表示带来的潜在性能上限，AR1 还支持直接对跨多个时间步的所有摄像头观测序列进行联合编码. 论文以 Flex 为例，该方法通过全自注意力层和一组固定的 query 向量对来自多摄像头、多时间步的图像 token 进行压缩，从而显式控制信息大小. 相比单图像编码，该方法可实现最高 20 倍的 token 压缩率，同时驾驶指标持平甚至略有提升.

## 轨迹的双重表征

AR1 对轨迹同时维护离散与连续两套表征. 序列 $u = \{(a^i, \kappa^i)\}_{i=1}^{64}$ 一方面被均匀量化为 128 个 special tokens，拼接在推理文本之后以 next-token prediction 目标训练 VLM；另一方面经正弦位置编码与 MLP 投影映射到 VLM 嵌入空间，送入独立的 action-expert. 该模块以 VLM 的 KV-cache 为条件，通过流匹配学习去噪向量场（两条路径之间通过 stop-gradient 解耦）. 推理时只走连续路径：action-expert 从标准高斯噪声出发，经约 10 步 Euler 积分直接解码出连续轨迹，不再生成离散 token.
                                                
这一设计背后有四重考量：
                          
1. 离散 token 使轨迹与推理文本共享同一序列空间，让 VLM 能在自回归框架内自然地耦合因果推理与驾驶行为；         
2. 离散表示为后训练阶段提供了直接的梯度流，使 GRPO 等策略梯度方法可以端到端地优化推理质量与推理-行动一致性；
3. 量化后的 token 为车辆动力学学习提供了强监督信号，而 flow-matching expert 则负责保证输出的连续性与多峰性；      
4. flow-matching 解码仅需约 10 步去噪，远快于自回归逐一采样 128 个 token，从而满足实时性要求；

## 强化学习

### 优化算法

AR1 使用 GRPO 更新模型参数，其目标函数为：

$$L_{\text {GRPO}}(\theta) = -\mathbb E_{\tau_i \sim \pi_\theta} \left[\dfrac {\exp (\beta A_i)}{\displaystyle \sum_j \exp(\beta A_j)} \log \pi_\theta (\tau_i) - \lambda_{\text {KL}} D_{\text {KL}} \left[\pi_\theta (\tau_i) \| \pi_{\text {ref}} (\tau_i)\right]\right], \quad A_i = r_i - \bar r \tag 2$$

该式存在两个关键设计：

1. 执行严格的 on policy 采样，故无需 clip 新旧策略比；
2. 不使用标准差做缩放，而是利用 softmax 光滑的优势函数直接作为对数概率的加权因子. $\beta$ 控制权重分布的尖锐程度，高 $\beta$ 让最优 rollout 获得更集中的关注，低 $\beta$ 保持更均匀的更新；

### 奖励建模

AR1 的奖励函数由三个信号构成，它们不是并列关系，而是一条层次因果链：

$$r = r_{\text{reason}} + r_{\text{consistency}} + r_{\text{traj}} \tag 3$$

第一层奖励 $r_{\text{reason}}$ 针对推理质量本身. 评判者是一个大型推理模型（如 DeepSeek-R1），输入包含：历史窗口最后一帧多相机视觉观测、来自数据集的 ground-truth reasoning trace、以及当前策略生成的预测 trace. LRM 根据两个维度来评判 rollout 推理质量好坏：

1. 纵向/横向驾驶意图（见论文 table 1）是否与 GT 匹配；
2. 历史观测中是否正确识别并引用了驱动该决策的 critical components（见论文 table 2）；

第二层奖励 $r_{\text{consistency}}$ 量化模型言行一致的程度. 论文描述的计算流程是：预测轨迹 → meta-actions，reasoning trace → 解析驾驶意图，然后 rule-based 匹配. 但论文未说明两侧如何对齐到同一表示空间——Table 5 的 meta-actions 是帧级运动学原语，Table 1 的驾驶决策是片段级语义，二者之间存在层次 gap.

第三层奖励 $r_{\text{traj}}$ 旨在确保实际连续轨迹贴合专家演示，并通过惩罚碰撞与急刹以确保安全性与舒适性：

$$r_{\text{traj}} = \lambda_{L_2} \|x_{\text{pred}} - x_{\text{expert}}\|^2_2 + \lambda_{\text{coll}} \mathbb I [\text{collision}(x_{\text{pred}})] + \lambda_{\text{jerk}} J(x_{\text{pred}}) \tag 4$$

### 数据筛选

与 SFT 直接在标注数据上计算 loss 不同，RL 的 on-policy 采样和 reward 调用会大幅放大计算成本. 论文因此提出用自身模型概率计算与外部模型显式奖励的分歧度来筛选训练数据，以最小的计算代价获得最大的对齐收益. 论文将这批高分歧样本与相近比例的随机采样数据混合，构成最终的 RL 训练集：前者保证对齐效率，后者维持分布多样性.