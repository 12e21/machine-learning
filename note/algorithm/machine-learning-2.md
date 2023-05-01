# 2.1 模型描述

- 在本节中，我们将介绍如何使用线性回归模型来表示数据之间的关系，并预测连续值的输出。
- 线性回归模型的一般形式为：$h_\theta(x) = \theta_0 + \theta_1 x$，其中$h_\theta(x)$表示假设函数，$\theta_0$和$\theta_1$表示模型参数，$x$表示输入变量。
- 我们的目标是通过训练数据集来找到最合适的参数$\theta_0$和$\theta_1$，使得假设函数能够尽可能地拟合数据，并且最小化预测误差。
- 为了表示多个输入变量的情况，我们可以将$x$和$\theta$都扩展为向量，并使用矩阵乘法来简化模型表示：$h_\theta(x) = \theta^T x$，其中$x_0 = 1$。
- 例如，如果我们有两个输入变量$x_1$和$x_2$，那么我们可以写成：$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 = \begin{bmatrix} \theta_0 & \theta_1 & \theta_2 \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} = \theta^T x$

# 2.2 代价函数

- 在本节中，我们将介绍如何定义和计算代价函数，以评估线性回归模型的预测误差，并找到最优的模型参数。
- 代价函数（cost function）又称为损失函数（loss function），是一个衡量模型预测值与真实值之间差异的函数，通常用平均平方误差（mean squared error）来表示。
- 代价函数的一般形式为：$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$，其中$m$表示样本数量，$h_\theta(x^{(i)})$表示第$i$个样本的预测值，$y^{(i)}$表示第$i$个样本的真实值。
- 我们的目标是通过优化算法（如梯度下降法）来找到最小化代价函数的模型参数$\theta$，从而得到最佳的线性回归模型。
- 为了方便计算和表示，我们可以将$x$和$\theta$都扩展为向量，并使用矩阵乘法来简化代价函数：$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$，其中$X$表示输入矩阵，$y$表示输出向量。
- 例如，如果我们有两个输入变量$x_1$和$x_2$，那么我们可以写成：$J(\theta) = \frac{1}{2m} (\begin{bmatrix} 1 & x_1^{(1)} & x_2^{(1)} \\ \vdots & \vdots & \vdots \\ 1 & x_1^{(m)} & x_2^{(m)} \end{bmatrix} \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \end{bmatrix} - \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(m)} \end{bmatrix})^T (\begin{bmatrix} 1 & x_1^{(1)} & x_2^{(1)} \\ \vdots & \vdots & \vdots \\ 1 & x_1^{(m)} & x_2^{(m)} \end{bmatrix} \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \end{bmatrix} - \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(m)} \end{bmatrix})$


- 我们就可以将代价函数看作是一个关于$\theta_0$和$\theta_1$的二元函数，并绘制出它的三维图像或者等高线图。
- 从三维图像中，我们可以看到代价函数是一个凸函数（convex function），也就是说它只有一个全局最小值点，没有局部最小值点。这样就保证了我们可以通过优化算法找到全局最优解。
- 从等高线图中，我们可以看到代价函数在不同的$\theta_0$和$\theta_1$取值下有不同的高度（也就是误差大小）。我们想要找到使得高度最低（也就是误差最小）的$\theta_0$和$\theta_1$组合。

# 2.3 梯度下降

## 什么是梯度下降

- 梯度下降是一种优化算法，它可以用来找到一个函数的最小值（或者最大值）。梯度下降的思想是，从一个初始点开始，沿着函数的负梯度方向（或者正梯度方向）移动一小步，然后重复这个过程，直到达到一个局部最小值（或者局部最大值）。

- 梯度是一个向量，它表示了函数在某一点的方向导数，也就是函数在不同方向上变化的快慢。梯度的大小表示了函数变化的速率，梯度的方向表示了函数变化最快的方向。因此，沿着负梯度方向移动，可以使得函数值下降最快。

## 梯度下降的公式

假设我们有一个目标函数 $J(\theta)$ ，它是关于参数 $\theta$ 的函数，我们想要找到使得 $J(\theta)$ 最小的 $\theta$ 。那么，我们可以用以下的公式来更新 $\theta$ ：

$$
\theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta}
$$

其中， $\alpha$ 是一个正数，叫做学习率（learning rate），它决定了每一步移动的大小。 $\frac{\partial J(\theta)}{\partial \theta}$ 是 $J(\theta)$ 关于 $\theta$ 的偏导数（partial derivative），它表示了 $J(\theta)$ 在 $\theta$ 方向上的梯度。

如果 $\theta$ 是一个多维向量，那么我们需要对每个分量都进行更新：

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

其中， $j$ 表示第 $j$ 个分量。注意，这里的更新需要同时进行，也就是说，在计算新的 $\theta_j$ 时，不能使用已经更新过的 $\theta_i$ ，而要使用原来的 $\theta_i$ 。

## 线性回归的梯度下降

- 假设我们有一个线性回归问题，我们想要找到一条直线来拟合数据点 $(x^{(i)}, y^{(i)})$ ，其中 $i = 1, 2, ..., m$ 。我们可以定义目标函数为：

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$

其中，

$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

是我们的假设函数（hypothesis function），它表示了直线的方程。我们想要找到使得 $J(\theta_0, \theta_1)$ 最小的 $\theta_0$ 和 $\theta_1$ 。那么，我们可以用梯度下降算法来更新 $\theta_0$ 和 $\theta_1$ ，具体如下：

- 初始化 $\theta_0$ 和 $\theta_1$ 为任意值（比如0）。
- 重复以下步骤，直到收敛或达到最大迭代次数：
  - 计算梯度： $\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$ 和 $\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$ 。
  - 更新参数： $\theta_0 := \theta_0 - \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0}$ 和 $\theta_1 := \theta_1 - \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1}$ 。

- 这种梯度下降称作Batch,因为每次计算误差都需要计算训练集容量M次