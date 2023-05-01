# 6.1 分类

分类是一种监督学习，它的目标是根据输入特征预测输出变量的离散值，例如类别标签。分类问题的例子有：

- 垃圾邮件过滤：根据邮件内容判断是否是垃圾邮件。
- 图像识别：根据图像像素判断图像中包含哪些物体。
- 医疗诊断：根据病人的症状和检查结果判断病人是否患有某种疾病。

分类问题可以分为二元分类和多元分类。二元分类是指输出变量只有两个可能的值，例如是或否，正或负，0或1等。多元分类是指输出变量有多于两个可能的值，例如动物的种类，颜色的名称，电影的类型等。

分类问题的常用算法有：

- 逻辑回归：使用一个对数几率函数（logistic function）或者称为 Sigmoid 函数来建立输入特征和输出变量之间的非线性关系，并使用最大似然估计（maximum likelihood estimation）来优化模型参数。
- 决策树：使用一系列的判断规则（if-then-else）来划分输入特征空间，并在每个叶节点上给出一个输出变量的值或概率分布。决策树可以使用信息增益（information gain），基尼不纯度（Gini impurity）或者其他指标来选择最优的划分属性和阈值。
- 支持向量机（SVM）：使用一个超平面（hyperplane）或者一个核函数（kernel function）来将输入特征空间映射到一个高维空间，并在该空间中寻找一个最大间隔（maximum margin）来分隔不同类别的数据点。SVM 可以使用硬间隔（hard margin）或者软间隔（soft margin）来处理线性可分或者线性不可分的情况，并可以使用拉格朗日乘子法（Lagrange multiplier method）或者序列最小优化算法（sequential minimal optimization algorithm）来优化模型参数。
- K 近邻（KNN）：使用一个距离度量（distance metric），例如欧几里得距离（Euclidean distance），曼哈顿距离（Manhattan distance）或者余弦相似度（cosine similarity）来计算输入特征与训练数据中每个数据点的相似度，并根据最近的 K 个邻居的输出变量的值或概率分布来预测输入特征的输出变量。KNN 是一种惰性学习（lazy learning），它不需要训练模型参数，但需要存储所有的训练数据，并在预测时进行计算。
- 朴素贝叶斯（Naive Bayes）：使用贝叶斯定理（Bayes' theorem）来计算输入特征在不同类别下的条件概率，并根据最大后验概率（maximum posterior probability）来预测输入特征的输出变量。


# 6.2 假设陈述

逻辑回归是一种常用的分类算法，它的假设函数是一个对数几率函数（logistic function），也称为 Sigmoid 函数，它的形式如下：

$$h_\theta(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}$$

其中，$g(z)$ 是 Sigmoid 函数，它的图像如下：

![Sigmoid function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png)

Sigmoid 函数的特点是它的输出值在 0 到 1 之间，可以表示一个概率值。我们可以将 $h_\theta(x)$ 解释为给定输入特征 $x$ 时，输出变量 $y$ 为 1 的概率，即：

$$h_\theta(x) = P(y=1|x;\theta)$$

因此，我们可以根据 $h_\theta(x)$ 的值来预测 $y$ 的值，例如：

$$y = \begin{cases} 1 & \text{if } h_\theta(x) \geq 0.5 \\ 0 & \text{if } h_\theta(x) < 0.5 \end{cases}$$

这相当于使用一个阈值（threshold）来划分两个类别。注意，当 $\theta^Tx \geq 0$ 时，$h_\theta(x) \geq 0.5$；当 $\theta^Tx < 0$ 时，$h_\theta(x) < 0.5$。因此，我们也可以使用 $\theta^Tx = 0$ 来划分两个类别。这相当于使用一个超平面（hyperplane）来分隔输入特征空间。


# 6.3 决策边界

- 决策边界（decision boundary）是指在特征空间中，能够将不同类别的样本分开的边界。在逻辑回归中，决策边界是由假设函数 $h_\theta(x)$ 确定的，它是满足 $\theta^Tx = 0$ 的所有点的集合。例如，如果我们有两个特征 $x_1$ 和 $x_2$，并且参数 $\theta$ 是向量 $[-3, 1, 1]$，那么决策边界是直线 $x_1 + x_2 = 3$，它将平面分为两个区域，一个区域预测 $y = 1$，另一个区域预测 $y = 0$。


- 决策边界不一定是线性的，也可以是非线性的。例如，如果我们有两个特征 $x_1$ 和 $x_2$，并且参数 $\theta$ 是向量 $[-1, 0, 0, 1, 1]$，那么决策边界是圆形 $x_1^2 + x_2^2 = 1$，它将平面分为两个区域，一个区域预测 $y = 1$，另一个区域预测 $y = 0$。


- 决策边界的形状取决于我们选择的特征和参数。我们可以通过增加更多的特征或使用高阶多项式特征来得到更复杂的决策边界。


# 6.4 代价函数

代价函数（cost function）是用来衡量模型预测的准确性的函数，它是模型参数 $\theta$ 的函数，表示在给定 $\theta$ 的情况下，模型对训练集的预测与真实标签之间的误差。我们的目标是找到使代价函数最小化的 $\theta$，这样就能得到最优的模型。

在逻辑回归中，我们不能使用线性回归中的均方误差（mean squared error）作为代价函数，因为这会导致代价函数是 $\theta$ 的非凸函数（non-convex function），即有很多局部最小值（local minima），这样就不容易找到全局最小值（global minimum）。

为了解决这个问题，我们使用另一种形式的代价函数，称为对数似然损失（log-likelihood loss），它是 $\theta$ 的凸函数（convex function），即只有一个最小值。

对数似然损失的定义如下：

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)}))]
$$

其中 $m$ 是训练集的样本数，$y^{(i)}$ 是第 $i$ 个样本的真实标签，$h_\theta(x^{(i)})$ 是第 $i$ 个样本的预测概率。

这个代价函数的直观解释是：如果 $y^{(i)} = 1$，那么我们希望 $h_\theta(x^{(i)})$ 越接近 1 越好，这样就能使 $-\log h_\theta(x^{(i)})$ 越小；如果 $y^{(i)} = 0$，那么我们希望 $h_\theta(x^{(i)})$ 越接近 0 越好，这样就能使 $-\log (1 - h_\theta(x^{(i)}))$ 越小。

我们可以用梯度下降法（gradient descent）或其他高级优化方法（advanced optimization methods）来求解使代价函数最小化的 $\theta$。

# 6.5 梯度下降法

梯度下降法（gradient descent）是一种求解最优化问题的迭代算法，它的基本思想是：从一个初始点开始，沿着函数的负梯度方向（即下降最快的方向）移动一定的步长，然后重复这个过程，直到达到一个局部最小值（local minimum）或者收敛。

在逻辑回归中，我们可以用梯度下降法来求解使代价函数 $J(\theta)$ 最小化的参数 $\theta$。具体的步骤如下：

1. 初始化 $\theta$ 为一个随机向量（或者全零向量）。
2. 计算代价函数 $J(\theta)$ 和梯度 $\frac{\partial J(\theta)}{\partial \theta}$, 其中梯度计算公式为$\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$。
3. 更新 $\theta$ 为 $\theta - \alpha \frac{\partial J(\theta)}{\partial \theta}$，其中 $\alpha$ 是学习率（learning rate），控制每次迭代的步长。
4. 重复步骤 2 和 3，直到 $\theta$ 收敛或者达到最大迭代次数。

梯度下降法的优点是简单易实现，适用于大规模数据集；缺点是可能陷入局部最小值，或者收敛速度较慢。

# 6.6 高级优化方法

高级优化方法（advanced optimization methods）是一些可以替代梯度下降法的算法，它们可以更快地找到最优解，或者更容易地逃离局部最小值。一些常见的高级优化方法有：

- 共轭梯度法（conjugate gradient method）
- BFGS算法（Broyden–Fletcher–Goldfarb–Shanno algorithm）
- L-BFGS算法（limited-memory BFGS algorithm）

这些算法的原理比较复杂，不需要我们深入了解，只需要知道它们的输入和输出即可。它们的输入是一个代价函数 $J(\theta)$ 和一个梯度函数 $\frac{\partial J(\theta)}{\partial \theta}$，它们的输出是一个使 $J(\theta)$ 最小化的 $\theta$。

# 6.7 多元分类

- 多元分类（multi-class classification）是指将样本分为多个类别的问题，例如，将手写数字识别为0-9的10个类别。

- 对于多元分类问题，我们可以使用一对多（one-vs-all）的策略，即针对每个类别训练一个逻辑回归分类器，将该类别的样本视为正例，其他类别的样本视为反例。然后，对于一个新的输入样本，我们计算每个分类器的预测概率，并选择概率最大的那个类别作为最终的预测结果。