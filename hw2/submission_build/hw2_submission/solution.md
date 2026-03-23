# Homework 2 解答

## 运行环境

- `Python 3.12`
- 依赖安装：

```bash
cd hw2
python -m venv .venv
.venv\Scripts\python -m pip install numpy matplotlib ir_sim==1.1.8
```

## Question 1

### (a) Imagine you want your robot to perform navigation tasks, which approach would you choose?

我会选择分层的混合式导航架构。高层使用规划方法给出全局目标和大致路径，低层使用 behavior-based / reactive 方法处理局部避障、实时响应和执行控制。  
原因是纯规划方法对模型和地图依赖较强，而纯反应式方法又容易缺少全局最优性；混合架构可以同时兼顾全局性和实时性。

### (b) What are the benefits of the behavior based paradigm?

- 实时性强，能够直接根据传感器输入快速产生控制量。
- 模块化好，不同行为可以独立设计后再组合。
- 对环境不确定性更鲁棒，不必依赖完全精确的世界模型。
- 对地图依赖较低，适合动态环境和部分未知环境。
- 易于并行组织多个行为，例如趋近目标、避障、沿边运动等。

### (c) Which approaches will win in the long run?

长期来看，单一范式通常都不是最终赢家，真正有竞争力的是混合架构。未来更可能胜出的是“规划 + 反应式控制 + 学习方法”的融合系统：  
规划提供全局结构，行为式方法提供局部实时响应，学习方法负责复杂环境中的参数适应与策略改进。

## Question 2

### (a) 五种力场的数学表达

记机器人位置为

$$
\mathbf{x}=\begin{bmatrix}x\\y\end{bmatrix},
$$

目标点为

$$
\mathbf{g}=\begin{bmatrix}g_x\\g_y\end{bmatrix},
$$

点障碍物为

$$
\mathbf{o}=\begin{bmatrix}o_x\\o_y\end{bmatrix}.
$$

若线段障碍物的两个端点分别为 $\mathbf{v}, \mathbf{w}$，则机器人到线段的最近点可写为

$$
t=\mathrm{clip}\left(\frac{(\mathbf{x}-\mathbf{v})^\top(\mathbf{w}-\mathbf{v})}{\|\mathbf{w}-\mathbf{v}\|^2}, 0, 1\right),
$$

$$
\mathbf{p}_{proj}=\mathbf{v}+t(\mathbf{w}-\mathbf{v}).
$$

在实现中取 $\varepsilon=10^{-6}$，并对需要时的速度范数裁剪到 `2.0`。

#### 1. Uniform force

$$
\mathbf{F}_u = k_u \frac{\mathbf{v}}{\|\mathbf{v}\|+\varepsilon}
$$

其中默认 $k_u=1.0$，方向固定为 `[[1], [0]]^T`。

#### 2. Perpendicular force

$$
\mathbf{F}_p = k_p \frac{\mathbf{x}-\mathbf{p}_{proj}}{\|\mathbf{x}-\mathbf{p}_{proj}\|+\varepsilon}
$$

其中默认 $k_p=1.2$，表示机器人沿障碍物法向远离线段。

#### 3. Attractive force

$$
\mathbf{F}_a = k_a (\mathbf{g}-\mathbf{x})
$$

其中默认 $k_a=0.6$，并对结果做最大范数裁剪，防止速度过大。

#### 4. Repulsive force

当障碍物距离 $d=\|\mathbf{x}-\mathbf{o}\|$ 小于作用半径 $d_0$ 时，

$$
\mathbf{F}_r = k_r\left(\frac{1}{d}-\frac{1}{d_0}\right)\frac{1}{d^2}\frac{\mathbf{x}-\mathbf{o}}{\|\mathbf{x}-\mathbf{o}\|+\varepsilon},
\quad d < d_0
$$

否则

$$
\mathbf{F}_r = \mathbf{0}.
$$

代码中默认 $k_r=1.0$，$d_0=2.0$。

#### 5. Tangential force

设逆时针旋转矩阵为

$$
\mathbf{R}_{90}=
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix},
$$

则切向力定义为

$$
\mathbf{F}_t = k_t \mathbf{R}_{90}\frac{\mathbf{x}-\mathbf{o}}{\|\mathbf{x}-\mathbf{o}\|+\varepsilon}
$$

其中默认 $k_t=1.0$，方向固定为逆时针。

### (b) 仿真结果

代码实现位于 `source/potential_fields.py`，运行命令如下：

```bash
cd hw2/source
..\.venv\Scripts\python question2_run.py -f uniform
..\.venv\Scripts\python question2_run.py -f perpendicular
..\.venv\Scripts\python question2_run.py -f attractive
..\.venv\Scripts\python question2_run.py -f repulsive
..\.venv\Scripts\python question2_run.py -f tangential
```

- `uniform`：机器人沿固定方向做近似匀速直线运动。  
  ![uniform](animation/uniform.gif)
- `perpendicular`：机器人沿线障碍法向远离障碍物。  
  ![perpendicular](animation/perpendicular.gif)
- `attractive`：机器人朝目标点收敛。  
  ![attractive](animation/attractive.gif)
- `repulsive`：机器人远离给定点障碍。  
  ![repulsive](animation/repulsive.gif)
- `tangential`：机器人围绕点障碍做逆时针绕行。  
  ![tangential](animation/tangential.gif)

## Question 3

### 思路

U 形障碍会导致标准势场在内部产生局部极小值，因此我在 `source/question3_run.py` 中采用了三层策略：

1. 保留目标吸引力：

$$
\mathbf{F}_{att}=k_a(\mathbf{g}-\mathbf{x}), \quad k_a=0.5
$$

2. 对三条线段分别使用“最近点排斥力 + 两侧不同增益”：

- 底边：内侧 `0.25`，外侧 `1.2`
- 右边：内侧 `1.4`，外侧 `0.2`
- 顶边：内侧 `0.9`，外侧 `0.3`
- 统一作用半径：`1.5`

3. 当连续 `15` 步合力范数小于 `0.05` 时，判定为陷入局部陷阱：

- 先叠加 `10` 步左向逃逸匀速场

$$
\mathbf{F}_{escape}=0.6
\begin{bmatrix}
-1\\
0
\end{bmatrix}
$$

- 同时把临时脱困目标切换到

$$
\mathbf{g}_{escape}=
\begin{bmatrix}
4.5\\
2.7
\end{bmatrix}
$$

- 当机器人运动到 U 形开口外侧（`x < 4.8` 且 `y < 3.0`）后，再恢复真实目标 `[[9], [2]]^T`

这样做的原因是：只靠左向匀速场虽然能把机器人推出一点，但在 `ir_sim 1.1.8` 里它会再次被真实目标吸回 U 形开口；增加一个临时脱困目标后，机器人会先绕出障碍，再重新朝真实目标前进。

### 仿真结果

运行命令：

```bash
cd hw2/source
..\.venv\Scripts\python question3_run.py
..\.venv\Scripts\python question3_run.py -a
```

验证结果：机器人可以脱离 U 形局部陷阱，到达目标点，且验证过程中未出现碰撞。

![potential_field](animation/potential_field.gif)

## 代码文件

- `source/potential_fields.py`：实现 `uniform / perpendicular / attractive / repulsive / tangential`
- `source/question3_run.py`：实现 Q3 的组合力场与脱困逻辑
