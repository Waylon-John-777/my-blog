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
```
```text
556 ms ± 1.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
```python
%timeit y = jax.jit(test_fn)(x).block_until_ready()
```
```text
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
对于多元向量函数 $g: \mathbb R^n \rightarrow \mathbb R^m$，我们可以使用 ```jax.jacobian``` 查看其雅可比矩阵. 此外，**JAX 的各种变换在遵循一定规则的情况下可以自由嵌套使用，这种可组合性是 JAX 区别于其它框架的重要特性**.

### jax.vmap
> ```jax.vmap```：自动将针对单个样本编写的函数向量化为批处理版本，无需手动编写循环或修改代码

## 分布式计算

## 显式随机管理