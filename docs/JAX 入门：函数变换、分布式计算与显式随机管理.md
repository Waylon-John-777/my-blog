# JAX 入门：函数变换、分布式计算与显式随机管理

*本文撰写于 2025 年 10 月 30 日，最后更新于 2025 年 11 月 19 日*

## 简介

<mark>**JAX 是 Google 开发的高性能数值计算库，主要用于机器学习研究**</mark>. 2025 年在旧金山的 PyTorch 会议中，Luca Antiga 在接受采访的时候明确指出 JAX 是 PyTorch 竞争力强大的对手之一：

> …“Throughout all these revolutions that came, you always see PyTorch there,” he said. “And there are, of course, others like JAX and so on —— they’re very strong.”…

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX1.png"><br>
  <b>Fig 1. Luca Antiga 是 PyTorch 原始论文的 21 位作者之一</b>
</div>

**Google 的大模型 Gemini 在最初的技术报告中也明确了模型的训练使用了 JAX 框架.**

> …The “single controller” programming model of JAX and Pathways allow a single Python process orchestrate the entire training run, dramatically simplifying the development workflow…

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX2.png"><br>
  <b>Fig 2. 2025 年 11 月 19 日发布的 Gemini 3 在主流评测集的分数实现了近乎全面（甚至是断崖式的）领先</b>
</div>

**这篇文档旨在通过尽可能简单的示例展现 JAX 有别于其它深度学习框架的特点，<mark>包括其函数式编程，分布式计算以及显式随机管理</mark>**. JAX 通过将 NumPy 的易用性、自动微分的强大以及 XLA 的极致性能相结合，为研究人员和开发者提供了一个强大的工具，用于探索机器学习和科学计算的前沿.

## 函数式编程

JAX 的核心能力在于对函数进行变换：

### jax.jit

> ```jax.jit```：即时编译，允许用户将函数编译成高效优化的版本，通常在加速代码运行方面非常有效

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX3.png"><br>
  <b>Fig 3. JAX 编译过程</b>
</div>

**Tracing**：JAX 不直接执行 Python 代码，而是用抽象值替代真实数据来追踪函数的执行路径：

```python
def trace_demo(u, v):
    print(f"Tracing with u = {u}, v = {v}")
    w = u + v
    x = w ** 2
    print(f"Inside computation x = {x}")
    return x
```
```python
print("=== First time ===")
x = jax.jit(trace_demo)(u=1, v=2)
```
```text
=== First time ===
Tracing with u = JitTracer<~int32[]>, v = JitTracer<~int32[]>
Inside computation x = JitTracer<~int32[]>
```
```python
print("=== Second time ===")
x = jax.jit(trace_demo)(u=1, v=2)
```
```text
=== Second time ===
```

**Jaxpr（JAX expression）**：JAX 的中间表示语言，典型示例如下：

```python
print(jax.make_jaxpr(trace_demo)(u=1, v=2))
```
```text
{ lambda ; a:i32[] b:i32[]. let
    c:i32[] = add a b
    d:i32[] = integer_pow[y=2] c
  in (d,) }
```

**XLA（Accelerated Linear Algebra）**：Google 开发的领域专用编译器，专门优化线性代数运算；

**HLO（High Level Operations）**：XLA 的中间表示，<mark>优化主要集中在这一层面：常量折叠，算子融合等</mark>；

如下是使用 ```jax.jit``` 进行加速的一个简单示例：

```python
def test_fn(x):
	for _ in range(100):
		x = jnp.dot(x, x.T)
		x = jnp.tanh(x)
	return x
```
```python
%timeit y = test_fn(x).block_until_ready()
%timeit y = jax.jit(test_fn)(x).block_until_ready()
```
```text
556 ms ± 1.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
487 ms ± 2.28 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

其中 ```.block_until_ready()``` 用以<mark>确保异步执行完成</mark>，获得准确计时.

### jax.grad
> ```jax.grad```：JAX 的自动微分，用以计算函数的梯度

多元函数 $f: \mathbb R^n \rightarrow \mathbb R$ 可在扰动点附近做二阶展开：

$$f(\mathbf x + \mathbf {dx}) \approx f(\mathbf x) + \mathbf {dx} \cdot \nabla f(\mathbf x) + \frac{1}{2} \mathbf {dx} \mathbf {dx} : \nabla \nabla f(\mathbf x)\tag 1$$

```python
def f(t):
    x, y, z = t
    return jnp.sin(x) * jnp.cos(y) * z
    
def df(t):
    x, y, z = t
    return jnp.array(
        [
            jnp.cos(x) * jnp.cos(y) * z,
            -jnp.sin(x) * jnp.sin(y) * z,
            jnp.sin(x) * jnp.cos(y)
        ]
    )
    
def ddf(t):
    x, y, z = t
    return jnp.array(
        [
            [-jnp.sin(x) * jnp.cos(y) * z, -jnp.cos(x) * jnp.sin(y) * z, jnp.cos(x) * jnp.cos(y)],
            [-jnp.cos(x) * jnp.sin(y) * z, -jnp.sin(x) * jnp.cos(y) * z, -jnp.sin(x) * jnp.sin(y)],
            [jnp.cos(x) * jnp.cos(y), -jnp.sin(x) * jnp.sin(y), 0]
        ]
    )
```
```python
def all_close(a, b):
    return jax.tree.map(
    	partial(jnp.allclose, atol=1e-3, rtol=1e-3), a, b
    )
```
```python
t = jax.random.normal(jax.random.key(0), (3,))
print(f"Check the first order derivatives: {all_close(df(t), jax.grad(f)(t))}")
print(f"Check the second order derivatives: {all_close(ddf(t), jax.hessian(f)(t))}")
```
```text
Check the first order derivatives: True
Check the second order derivatives: True
```
此外，对于多元向量函数 $g: \mathbb R^n \rightarrow \mathbb R^m$，我们可以使用 ```jax.jacobian``` 查看其雅可比矩阵. <mark>**JAX 的各种变换在遵循一定规则的情况下可以自由嵌套使用，这种可组合性是 JAX 区别于其它框架的重要特性**</mark>.

### jax.vmap
> ```jax.vmap```：自动将针对单个样本编写的函数向量化为批处理版本，无需手动编写循环或修改代码

假设我们需要计算余弦相似度

```python
# Vector version
def cosine_simi(a, b):
	return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
	
# Loop version
def loop_cosine_simi(a, b):
	return jnp.array([cosine_simi(a, b) for a, b in zip(a, b)])
	
# Batch version
def batch_cosine_simi(a, b):
	return jnp.sum(a * b, axis=-1) / (jnp.linalg.norm(a, ord=2, axis=-1) * jnp.linalg.norm(b, ord=2, axis=-1))
	
# Prepare inputs
a = jax.random.normal(jax.random.key(0), shape=(1000, 256))
b = jax.random.normal(jax.random.key(1), shape=(1000, 256))
```
```python
%timeit loop_cosine_simi(a, b).block_until_ready()
%timeit batch_cosine_simi(a, b).block_until_ready()
%timeit jax.vmap(cosine_simi)(a, b).block_until_ready()
```
```text
40.3 ms ± 462 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)
149 μs ± 17.1 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
485 μs ± 8.66 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
可见 ```batch_cosine_simi``` 与 ```jax.vmap(cosine_simi)``` 性能相当（均为 μs 量级），<mark>**但 ```jax.vmap``` 的价值在于让复杂操作变得简单可写</mark>.**

除了上述提及的功能， JAX 中还存在：

1. ```jax.pmap```：并行映射，跨多设备并行执行；
2. ```jax.checkpoint```：梯度检查点，用时间换空间；
3. …

等诸多函数变换，这些变换在规则下可以任意组合，使用户以优雅的代码风格实现高效的计算速度.

## 分布式计算

### 数据分片

分布式计算的核心概念之一是数据分片，它描述了数据在可用设备之间的布局方式. **JAX 的数据类型——不可变的 ```jax.Array``` 数组结构——设计初衷就是<mark>面向分布式数据与计算</mark>**.

每一个 ```jax.Array``` 都有一个关联的 ```jax.sharding.Sharding``` 对象，用来描述在全局数组中，每个设备应该负责哪一块数据. 当我们从零创建一个 ```jax.Array``` 时，也必须同时创建其对应的 ```Sharding```. 一般情况下，数组被划分在单一设备，使用 ```jax.device_put``` 可将数组重分片：

```python
arr = jnp.arange(32).reshape(4, 8)

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=("x", "y"))
sharded_arr = jax.device_put(arr, NamedSharding(mesh, P("x", "y")))

jax.debug.visualize_array_sharding(sharded_arr)
```
<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX4.png"><br>
  <b>Fig 4. 检查张量在各设备上的切分情况</b>
</div>

### jax.shard_map

<mark>JAX 的并行计算存在三种模式</mark>：

1. 通过 ```jax.jit``` 实现的自动管理：编译器会自动选择最优计算策略；
2. Explicit sharding：编译器必须严格遵守用户提供的分片方式，所以受到更强约束；
3. **完全手动分片：```jax.shard_map``` 允许你编写面向单个设备的代码，并显式指定通信操作；**

三者之间区别如下表所示：

<center>

| 模式 | 视角 | 显式分片 | 显式通信 |
|:-------:|:-------:|:-------:|:-------:|
| Auto  | Global  | ❌  | ❌  |
| Explicit  | Global  | ✅  | ❌  |
| Manual  | Per-device  | ✅  | ✅  | 

</center>

本篇博客主要介绍 ```jax.shard_map```，其主要包含三个参数：```mesh```, ```in_specs``` 与 ```out_specs```：

> **```in_specs```：在某个位置提及设备轴名称表示将相应参数数组轴沿该设备轴进行分片；若未提及轴名称则表示复制.**

> **```out_specs```：在某个位置提及设备轴名称表示沿相应位置轴拼接分片；若未提及设备轴名称则表明该设备轴上各输出值相等，仅需返回单一数值.**

如下是一个使用 ```jax.shard_map``` 进行计算的 Naive 例子：

```python
mesh = jax.make_mesh((4, 2), ("x", "y"))

a = jnp.arange(8 * 16.).reshape(8, 16)
b = jnp.arange(16 * 4.).reshape(16, 4)

@jax.shard_map(
    mesh=mesh,
    in_specs=(P("x", "y"), P("y", None)),
    out_specs=P("x", None)
)
def matmul_basic(a_block, b_block):
    # a_block: [2, 8]
    # b_block: [8, 4]
    tmp = jnp.dot(a_block, b_block)
    c_block = jax.lax.psum(tmp, "y")
    # c_block: [2, 4]
    return c_block

all_close(matmul_basic(a, b), a @ b)
```
```text
True
```

在函数 ```matmul_basic``` 中：

1. 矩阵 a 的维度 0 与维度 1 沿设备轴 x, y 进行分片，单设备上可见的 a 子矩阵大小为 [8 / 4, 16 / 2] = [2, 8]；
2. 矩阵 b 的维度 0 沿设备轴 y 进行分片，由于设备轴 x 未提及，故 b 的子矩阵沿该设备轴方向做复制，单设备上可见的 b 的子矩阵大小为 [16 / 2， 4] = [8， 4]；
3. 各设备进行矩阵乘法的 local 操作；
4. 通过 ```jax.lax.psum``` 沿设备轴 y 进行 all-reduce；
5. 通信确保沿设备轴 y 方向数据的一致性，故返回结果只需沿设备轴 x 方向进行拼接；

<mark>**将设备视作网格并建立张量维度到设备轴的映射是 JAX 分布式计算的核心特色之一**</mark>. 下图展示了在 LLM 训练中常见的并行训练方式的设备 & 逻辑 mesh 的设置：

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX5.png"><br>
  <b>Fig 5. 各并行策略下模型权重与数据在不同卡上的切分</b>
</div>

### 通信原语
JAX 的通信可通过 ```jax.lax.psum```, ```jax.lax.all_to_all``` 等 API 实现. 在 GPU 设备上，这些集合通信操作经过编译后最终会调用 NCCL.

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX6.png"><br>
  <b>Fig 6. 常见通信操作图解</b>
</div>

<mark>**JAX 与 Pytorch 在分布式训练设计上的另一核心区别在于：JAX 可以“自动微分”集合通信操作，并生成相应的反向传播规则**</mark>. 这种设计让实现复杂的分布式策略更加优雅和可维护.

另一在 JAX 中常使用的通信为 ```jax.lax.ppermute```，该操作需要指定设备轴及一组源索引、目标索引对. 这些索引代表局部数据沿该设备轴的坐标. ```jax.lax.ppermute``` 会将参数从每个源发送到对应的目的地. 

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX7.png"><br>
  <b>Fig 7. 通过 permute 操作实现 reduce-scatter</b>
</div>

**在流水线并行以及计算-通信掩藏中，```jax.lax.ppermute``` 扮演着重要角色**.

### Naive DP, FSDP, TP

这一小节我们将展示数据并行，完全分片数据并行以及张量并行的 Naive 实现方式，首先需要初始化一个 Toy NN 和 random input/output：

```python
def init_layer(key, d_in, d_out):
	k1, k2 = jax.random.split(key)
	W = jax.random.normal(k1, (d_in, d_out)) / jnp.sqrt(d_in)
	b = jax.random.normal(k2, (d_out,))
	return W, b
	
def init(key, layer_sizes, batch_size):
	key, *keys = jax.random.split(key, len(layer_sizes))
	params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))
	key, *keys = jax.random.split(key, 3)
	inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
	targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))
	return params, (inputs, targets)

def predict(params, inputs):
	for W, b in params:
		outputs = jnp.dot(inputs, W) + b
		inputs = jax.nn.relu(outputs)
	return outputs

def loss(params, batch):
	inputs, targets = batch
	predictions = predict(params, inputs)
	return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))
```
#### Data Parallelism

我们首先从最简单的数据并行出发：

```python
mesh = jax.make_mesh(axis_shapes=(8,), axis_names=("dp",))

@jax.shard_map(
	mesh=mesh,
	in_specs=(P(None), P("dp")),
	out_specs=P()
)
def loss_dp(params, local_batch):
	inputs, targets = local_batch
	predictions = predict(params, inputs)
	local_loss = jnp.mean(
		jnp.sum((predictions - targets) ** 2, axis=-1)
	)
	return jax.lax.pmean(local_loss, "dp")
```

1. 设置 DP 轴；
2. 将网络参数沿 DP 轴复制，将输入/目标张量的第 0 维度沿 DP 轴切分；
3. 正常前向传播；
4. 将 loss 值沿设备 DP 轴做 all-reduce（```jax.lax.pmean```）；

可以通过比较 ```loss``` 与 ```loss_dp``` 以及 ```jax.grad(loss)``` 与 ```jax.grad(loss_dp)``` 来验证 Naive DP 中前后向传播的正确性. <mark>**在 JAX 中集合通信操作是可微的，开发者仅需编写前向传播逻辑，反向传播的通信操作将自动生成，这极大简化了代码**</mark>.

```python
params, batch = init(
	key=jax.random.key(42),
	layer_sizes=[784, 128, 128, 128, 128, 128, 64],
	batch_size=32
)
print(
	all_close(
		loss(params, batch), loss_dp(params, batch)
	)
)
print(
	all_close(
		jax.grad(loss)(params, batch), jax.grad(loss_dp)(params, batch)
	)
)
```

```text
True
True
```
#### Fully Sharded Data Parallelism

完全分片数据并行是由 Meta 提出的分布式训练方式. 其核心思想可由如下伪代码表述：

```text
FSDP forward pass:
	for layer_i in layers:
		all-gather full weights for layer_i
		forward pass for layer_i
		discard full weights for layer_i
		
FSDP backward pass:
	for layer_i in layers:
		all-gather full weights for layer_i
		backward pass for layer_i
		discard full weights for layer_i
		reduce-scatter gradients for layer_i
```

在 JAX 中的实现可以归纳如下：

```python
mesh = jax.make_mesh(axis_shapes=(8,), axis_names=("fsdp",))

@jax.shard_map(mesh=mesh, in_specs=(P("fsdp"), P("fsdp")), out_specs=P())
@partial(jax.checkpoint, policy=lambda op, *_, **__: str(op) != "all_gather")
def loss_fsdp(params_frag, local_batch):
	inputs, targets = local_batch
	for W_frag, b_frag in params_frag:
		W = jax.lax.all_gather(W_frag, "fsdp", tiled=True)
		b = jax.lax.all_gather(b_frag, "fsdp", tiled=True)
		outputs = jnp.dot(inputs, W) + b
		inputs = jax.nn.relu(outputs)
	local_loss = jnp.mean(
		jnp.sum((outputs - targets) ** 2, axis=-1)
	)
	return jax.lax.pmean(local_loss, "fsdp")
```

1. 设置 FSDP 轴；
2. 将网络参数，输入/目标张量的第 0 维度沿 FSDP 轴切分；
3. 前向传播临时聚合网络参数，用完丢弃；
4. 沿 FSDP 轴对 loss 进行归约；

基于 ```all_close``` 不难对 ```loss_fsdp``` 进行正确性验证. **上述代码利用 ```jax.checkpoint``` 控制网络的中间缓存量，<mark>未有缓存的中间值将在反向传播时重新计算</mark>**.

#### Tensor Parallelism

最后我们简要阐述一下张量并行，张量并行的核心思想在于利用矩阵的分块乘法，其实现思路大致如下：

```python
mesh = jax.make_mesh(axis_shapes=(8,), axis_names=("tp",))

@jax.shard_map(
	mesh=mesh,
	in_specs=(P(None, "tp"), P("tp", None), P("tp")),
	out_specs=P(None, "tp")
)
def gemm_tp(inputs, W, b):
	outputs = jnp.dot(inputs, W)
	return jax.lax.psum_scatter(outputs, "tp", scatter_dimension=1, tiled=True) + b
	
def predict_tp(params, inputs):
	for W, b in params:
		outputs = gemm_tp(inputs, W, b)
		inputs = jax.nn.relu(outputs)
	return outputs
	
def loss_tp(params, batch):
	inputs, targets = batch
	predictions = predict_tp(params, inputs)
	return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))
```

1. 设置 TP 轴；
2. 在线性变换中，将输入张量的第 1 维度和矩阵 W 的第 0 维度沿设备划分；
3. local 矩阵乘计算完毕后使用 reduce-scatter 进行通信，此时每张卡拿到矩阵乘结果沿第 1 维度的切片，再与本地 bias 相加以及通过激活函数；
4. 重复上述操作完成前向传播；

#### 联合并行简单示例

尽管这篇文档的目的在于以尽可能简单的示例展现 JAX 有别于其它深度学习框架的特点，<mark>但在分布式计算中，若不谈及联合并行方式，便不足以展现 JAX 的真正强大</mark>. 这一小节我们以 FSDP + TP 为例进行说明.

```python
mesh = jax.make_mesh(axis_shapes=(4, 2), ("fsdp", "tp"))

@partial(jax.checkpoint, policy=lambda op, *_, **__: str(op) != "all_gather")
def predict_fsdp_tp(params_frag, inputs):
	for W_frag, b_frag in params_frag:
		W = jax.lax.all_gather(W_frag, "fsdp", tiled=True)
		b = jax.lax.all_gather(b_frag, "fsdp", tiled=True)
		outputs = jnp.dot(inputs, W)
		outputs = jax.lax.psum_scatter(outputs, "tp", scatter_dimension=1, tiled=True) + b
		inputs = jax.nn.relu(outputs)
	return outputs

@jax.shard_map(
	mesh=mesh,
	in_specs=(P(("tp", "fsdp")), P("fsdp", "tp")),
	out_specs=P()
)
def loss_fsdp_tp(local_params, local_batch):
	inputs, targets = local_batch
	predictions = predict_fsdp_tp(local_params, inputs)
	loss = jax.lax.psum(jnp.sum((predictions - targets) ** 2, axis=-1), "tp")
	return jax.lax.pmean(jnp.mean(loss), "fsdp")
```

1. 划分 4 * 2 设备 mesh，其中第 0 维是 FSDP 轴，第 1 维是 TP 轴；
2. 将输入/目标张量的维度 0 沿 FSDP 轴切片，维度 1 沿 TP 轴切片；
3. 将网络参数的第 0 维度沿 FSDP & TP 的联合轴做划分（列优先）；
4. 在前向传播中，通过 ```jax.lax.all_gather``` 指定网络参数沿 FSDP 轴做聚合（```jax.checkpoint``` 在当前层计算完成后放弃对 all-gather 结果的缓存）；
5. local 矩阵乘操作，沿 TP 轴做 reduce-scatter，再与本地 bias 相加；
6. 输出层计算损失时，先沿 TP 轴求和，再沿 FSDP 轴做平均；

## 显式随机管理

与 NumPy, PyTorch 等库不同，JAX 采用显式的随机状态管理. 显式管理让程序行为<mark>可预测，可追踪，可复现</mark>——代价则是需要手动管理 key 的传递和分割.

```python
key = jax.random.key(42)
key, k1, k2 = jax.random.split(key, 3)
x = jax.random.normal(k1, shape=(3, 3))
y = jax.random.normal(k2, shape=(3, 3))
print(f"{x = }")
print(f"{y = }")
``` 
```text
x = Array([[ 0.60576403,  0.7990441 , -0.908927  ],
       [-0.63525754, -1.2226585 , -0.83226097],
       [-0.47417238, -1.2504351 , -0.17678244]], dtype=float32)
y = Array([[ 0.4323065 ,  0.5872638 , -1.1416743 ],
       [-0.37379906, -0.19910173, -1.7271094 ],
       [-1.8330271 , -0.46168378, -0.03195509]], dtype=float32)
```

显式 key 是一个不可变的值，这避免了分布式计算下的全局状态同步问题，实现了场景的精确控制.

### nnx.Rngs

Flax NNX（Google 开发的 Flax 神经网络库的新一代 API）使用有状态的 ```nnx.Rngs``` 类来简化 JAX 对随机性的管理. 要创建一个 ```nnx.Rngs``` 对象，你只需在构造时为任意关键字参数传入一个整数或 ```jax.random.key```：

```python
rngs = nnx.Rngs(params=0, dropout=1)
nnx.display(rngs)
```
```text
Rngs( # RngState: 4 (24 B)
	params=RngStream( # RngState: 2 (12 B)
		tag='params',
		key=RngKey( # 1 (8 B)
			value=# jax.Array key<fry>()
			Array((), dtype=key<fry>) overlaying:
			[0 0]
			,
			tag='params',
		),
		count=RngCount( # 1 (4 B)
			value=<jax.Array(0, dtype=uint32)>,
			tag='params',
		),
	),
	dropout=RngStream( # RngState: 2 (12 B)
		tag='dropout',
		key=RngKey( # 1 (8 B)
			value=# jax.Array key<fry>()
			Array((), dtype=key<fry>) overlaying:
			[0 1]
			,
			tag='dropout',
		),
		count=RngCount( # 1 (4 B)
			value=<jax.Array(0, dtype=uint32)>,
			tag='dropout',
		),
	),
)
```

<mark>```nnx.Rngs```能够创建多个具有名字的 PRNG key 流. 要生成新的 key，可以访问其中的一个流并调用它的 ```__call__``` 方法</mark>. **此操作会使 count 自增但 key 属性本身不会发生改变**.

```python
params_key = rngs.params()
dropout_key = rngs.dropout()
print(f"{params_key = }")
print(f"{dropout_key = }")
```
```text
params_key = Array((), dtype=key<fry>) overlaying:
[1797259609 2579123966]
dropout_key = Array((), dtype=key<fry>) overlaying:
[ 507451445 1853169794]
```

<mark>**事实上 Flax NNX 内置的各类层只使用两个标准的 PRNG key 流名称：params 用于参数初始化，dropout 用于创建随机掩码**</mark>.

### 采样

在 JAX 中进行采样，一个常见的做法是生成一个 key 然后传给 ```jax.random``` 的某个函数. ```nnx.RngStream``` 提供了与 ```jax.random``` 相同签名的方法，但不再需要手动传递 key：

```python
x = rngs.params.normal((3, 3))
y = rngs.dropout.exponential((10,))
print(f"{x = }")
print(f"{y = }")
```
```text
x = Array([[-2.4424558 , -2.0356805 ,  0.20554423],
       [-0.3535502 , -0.76197404, -1.1785518 ],
       [-1.1482196 ,  0.29716578, -1.3105359 ]], dtype=float32)
y = Array([0.5169214 , 0.79605514, 0.2518507 , 0.19346236, 2.1336777 ,
       0.83475715, 0.10780825, 0.07331487, 0.5416562 , 1.9597657 ],      dtype=float32)
```

### 随机态分叉
假设我们在一个批次数据上使用 dropout，显然对于不同批次的 dropout mask 应使用不同的随机状态. 这可以通过 fork 方法实现：

```python
class Model(nnx.Module):
	def __init__(self, rngs: nnx.Rngs):
		self.linear = nnx.Linear(10, 10, rngs=rngs)
		self.dropout = nnx.Dropout(0.1)
		
	def __call__(self, x, rngs: nnx.Rngs):
		return nnx.relu(self.dropout(self.linear(x), rngs=rngs))
		
model = Model(rngs=nnx.Rngs(0))

@nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
def model_forward(model, x, rngs):
	return model(x, rngs=rngs)
	
forked_rngs = nnx.Rngs(1).fork(split=5)
model_forward(model, jnp.ones((5, 10)), forked_rngs)
```

在 Flax 中，还有另一种处理调用时随机性的方式：我们可以把随机状态直接绑定到 Module 内部，这样，随机状态就变成了另一类 Module State. 使用这种“隐式随机状态”需要在初始化 Module 时通过 rngs 关键字参数传入随机性：

```python
class Model(nnx.Module):
	def __init__(self, rngs: nnx.Rngs):
		self.linear = nnx.Linear(10, 10, rngs=rngs)
		self.dropout = nnx.Dropout(0.1, rngs=rngs)
		
	def __call__(self, x):
		return nnx.relu(self.dropout(self.linear(x)))
		
model = Model(rngs=nnx.Rngs(params=0, dropout=1))
```

## 参考资料
[[1] Manual parallelism with shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
[[2] Basic Guides: Randomness](https://flax.readthedocs.io/en/stable/guides/randomness.html)