# JAX 入门：函数变换、分布式计算与显式随机管理

*本文撰写于 2025 年 10 月 30 日，最后更新于 2025 年 11 月 19 日*

## 简介

**JAX 是 Google 开发的高性能数值计算库，主要用于机器学习研究**. 2025 年在旧金山的 PyTorch 会议中，Luca Antiga 在接受采访的时候明确指出 JAX 是 PyTorch 竞争力强大的对手之一：

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

**这篇文档旨在通过尽可能简单的示例展现 JAX 有别于其它深度学习框架的特点，包括其函数式编程，分布式计算以及显式随机管理**. JAX 通过将 NumPy 的易用性、自动微分的强大以及 XLA 的极致性能相结合，为研究人员和开发者提供了一个强大的工具，用于探索机器学习和科学计算的前沿.

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

**Jaxpr (JAX expression)**：JAX 的中间表示语言，典型示例如下：

```python
print(jax.make_jaxpr(trace_demo)(u=1, v=2))
```
```text
{ lambda ; a:i32[] b:i32[]. let
    c:i32[] = add a b
    d:i32[] = integer_pow[y=2] c
  in (d,) }
```

**XLA (Accelerated Linear Algebra)**：Google 开发的领域专用编译器，专门优化线性代数运算；

**HLO (High Level Operations)**：XLA 的中间表示，优化主要集中在这一层面：常量折叠，算子融合等；

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

其中 ```.block_until_ready()``` 用以确保异步执行完成，获得准确计时.

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
此外，对于多元向量函数 $g: \mathbb R^n \rightarrow \mathbb R^m$，我们可以使用 ```jax.jacobian``` 查看其雅可比矩阵. **JAX 的各种变换在遵循一定规则的情况下可以自由嵌套使用，这种可组合性是 JAX 区别于其它框架的重要特性**.

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
可见 ```batch_cosine_simi``` 与 ```jax.vmap(cosine_simi)``` 性能相当（均为 μs 量级），**但 ```jax.vmap``` 的价值在于让复杂操作变得简单可写.**

除了上述提及的功能， JAX 中还存在：

1. ```jax.pmap```：并行映射，跨多设备并行执行；
2. ```jax.checkpoint```：梯度检查点，用时间换空间；
3. …

等诸多函数变换，这些变换在规则下可以任意组合，使用户以优雅的代码风格实现高效的计算速度.

## 分布式计算

### 数据分片

分布式计算的核心概念之一是数据分片，它描述了数据在可用设备之间的布局方式. **JAX 的数据类型——不可变的 ```jax.Array``` 数组结构——设计初衷就是面向分布式数据与计算**.

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

JAX 的并行计算存在三种模式：

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

> **```in_specs```：在某个位置提及设备轴名称表示将相应参数数组轴沿该设备轴进行分片；若未提及轴名称则表示复制.
> ```out_specs```：在某个位置提及设备轴名称表示沿相应位置轴拼接分片；若未提及设备轴名称则表明该设备轴上各输出值相等，仅需返回单一数值.**

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
Array(True, dtype=bool)
```

在函数 ```matmul_basic``` 中：

1. 矩阵 a 的维度 0 与维度 1 沿设备轴 x, y 进行分片，单设备上可见的 a 子矩阵大小为 [8 / 4, 16 / 2] = [2, 8]；
2. 矩阵 b 的维度 0 沿设备轴 y 进行分片，由于设备轴 x 未提及，故 b 的子矩阵沿该设备轴方向做复制，单设备上可见的 b 的子矩阵大小为 [16 / 2， 4] = [8， 4]；
3. 各设备进行矩阵乘法的 local 操作；
4. 通过 ```jax.lax.psum``` 沿设备轴 y 进行 all-reduce；
5. 通信确保沿设备轴 y 方向数据的一致性，故返回结果只需沿设备轴 x 方向进行拼接；

**将设备视作网格并建立张量维度到设备轴的映射是 JAX 分布式计算的核心特色之一**. 下图展示了在 LLM 训练中常见的并行训练方式的设备 & 逻辑 mesh 的设置：

<div align="center">
  <img src="./figs/JAX 入门：函数变换、分布式计算与显式随机管理/JAX5.png"><br>
  <b>Fig 5. 各并行策略下模型权重与数据在不同卡上的切分</b>
</div>

## 显式随机管理