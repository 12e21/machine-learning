# 8.1 非线性假设

- 在一些机器学习问题中，线性假设可能不足以拟合数据，例如逻辑回归中的非线性分类问题。
- 为了解决这个问题，我们可以使用**多项式回归**，即在假设函数中添加更高次的特征项，例如$x^2$,$x^3$等。
- 多项式回归可以拟合更复杂的数据形状，但是也有一些缺点：
  - 需要选择合适的多项式次数，否则可能导致欠拟合或过拟合。
  - 需要对特征进行**特征缩放**，即将每个特征的值调整到相近的范围，以加快梯度下降的收敛速度。
  - 需要使用**正则化**来防止过拟合，即在代价函数中添加一个惩罚项，来减小高次特征的系数。

# 8.2 神经元与大脑

- 机器学习中的神经网络是受到生物神经系统的启发而设计的一种算法。
- 生物神经系统由许多**神经元**组成，每个神经元有一个**细胞体**和多个**树突**和**轴突**。
- 树突负责接收其他神经元传来的信号，细胞体负责对信号进行处理，轴突负责将处理后的信号传递给其他神经元。
- 一个神经元可以被看作是一个简单的逻辑单元，它根据输入信号的强度和阈值来决定是否激活输出信号。
- 机器学习中的神经网络是由许多类似于神经元的单元组成的，每个单元有多个输入和一个输出，输入和输出之间有一些权重参数和激活函数。
- 神经网络可以拟合非线性假设，而且可以自动地从数据中学习特征，而不需要人为地设计特征。


# 8.3 模型表示I

- 一个神经网络由多层单元组成，每一层有多个单元，每个单元有多个输入和一个输出。
- 第一层称为**输入层**，最后一层称为**输出层**，中间的层称为**隐藏层**。
- 输入层的单元对应于数据的特征，输出层的单元对应于数据的标签，隐藏层的单元对应于数据的隐含特征。
- 每一层的单元都与下一层的所有单元相连，每条连接都有一个权重参数，表示该连接的强度和方向。
- 每个单元都有一个激活函数，用于将输入信号转换为输出信号，常用的激活函数有**sigmoid函数**，**tanh函数**和**ReLU函数**等。
- 一个神经网络可以被看作是一个复合函数，它由多个简单函数组合而成，每个简单函数都由一层单元的输入、权重和激活函数决定。


# 8.4 模型表示II

- 为了方便地表示神经网络的结构和参数，我们可以使用一些符号和矩阵来表示。
- 我们用$L$表示神经网络的层数，用$s_l$表示第$l$层的单元数（不包括偏置单元），用$K$表示输出层的单元数。
- 我们用$a_i^{(l)}$表示第$l$层的第$i$个单元的激活值，用$\Theta^{(l)}$表示第$l$层到第$l+1$层的权重矩阵，其大小为$s_{l+1} \times (s_l + 1)$。
- 我们用$z_i^{(l)}$表示第$l$层的第$i$个单元的输入值，即$a_i^{(l)} = g(z_i^{(l)})$，其中$g$是激活函数。
- 我们用$x$表示输入层的激活值，即$a^{(1)} = x$，用$h_\Theta(x)$表示输出层的激活值，即$a^{(L)} = h_\Theta(x)$。
- 我们用向量化的方式来计算每一层的激活值，即$a^{(l+1)} = g(\Theta^{(l)}a^{(l)})$，其中$a^{(l)}$和$a^{(l+1)}$都需要添加一个偏置单元。

# 8.5 例子与直觉理解I

- 为了更好地理解神经网络的工作原理，我们可以看一些具体的例子和直觉图解。
- 一个简单的例子是使用一个三层的神经网络来实现逻辑运算，例如AND，OR，NOT和XOR等。
- 我们可以将每个逻辑运算看作是一个二分类问题，即给定两个输入$x_1$和$x_2$，输出$y$为0或1。
- 我们可以为每个逻辑运算设计一个合适的假设函数，即选择合适的权重矩阵$\Theta^{(1)}$和$\Theta^{(2)}$，使得$h_\Theta(x)$能够正确地输出$y$。
- 我们可以用图形的方式来表示每个逻辑运算的决策边界，即在平面上画出一条或多条直线，将不同类别的数据点分开。
- 我们可以发现，使用神经网络可以实现一些线性分类器无法实现的逻辑运算，例如XOR，因为神经网络可以拟合非线性的决策边界。

# 8.6 例子与直觉理解II

- 另一个例子是使用一个三层的神经网络来实现手写数字的识别，即给定一个28x28像素的图片，输出一个0到9的数字。
- 我们可以将每个像素看作是一个特征，即输入层有784个单元，输出层有10个单元，每个单元对应于一个数字。
- 我们可以为隐藏层选择合适的单元数，例如25个，这样隐藏层的每个单元都可以看作是一个小型的特征检测器，用于检测图片中的一些基本形状，例如边缘，角点，曲线等。
- 我们可以用图形的方式来表示隐藏层的激活值，即将每个单元的激活值映射到一个灰度图像，显示该单元检测到的特征。
- 我们可以发现，使用神经网络可以实现一些复杂的图像识别任务，因为神经网络可以自动地从数据中学习特征，而不需要人为地设计特征。

# 8.7 多类分类

- 在一些机器学习的问题中，我们需要对数据进行多类分类，即给定一个输入$x$，输出一个类别$y$，其中$y$可以取多个值，例如1到10。
- 我们可以使用神经网络来实现多类分类，只需要将输出层的单元数设置为类别的个数，即$K$个。
- 我们可以使用**one-hot编码**来表示输出层的标签，即用一个$K$维的向量来表示$y$，其中只有一个元素为1，其余为0，表示$y$属于哪个类别。
- 我们可以使用**交叉熵损失函数**来衡量神经网络的预测误差，即用$h_\Theta(x)$和$y$之间的距离来表示损失。
- 我们可以使用**梯度下降法**或其他优化算法来更新神经网络的参数，即根据损失函数的梯度来调整权重矩阵$\Theta^{(l)}$。
