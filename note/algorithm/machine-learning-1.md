#  第一部分：机器学习简介
  
  
##  1-1.欢迎参加《机器学习》课程
  
  
- 本课程由吴恩达教授主讲，介绍机器学习的基本概念和应用。
- 机器学习是人工智能的一个子领域，涉及让计算机从数据中自动学习和改进。
- 本课程将涵盖监督学习、无监督学习、神经网络、支持向量机、异常检测、推荐系统等主题。
- 本课程将使用Octave/Matlab作为编程语言，但不需要有太多的编程经验。
  
##  1-2.什么是机器学习？
  
  
- 机器学习有很多种定义，比如：
  
    - “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.” (Tom Mitchell, 1997)
  
    - “Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.” (Arthur Samuel, 1959)
  
- 根据是否给出正确答案，可以把机器学习分为两大类：
  
    - 监督学习（supervised learning）：给出输入和输出之间的对应关系，让计算机找出规律。例如：回归问题（预测连续值）、分类问题（预测离散值）。
  
    - 无监督学习（unsupervised learning）：只给出输入数据，没有给出输出或标签，让计算机自己发现数据中的结构或模式。例如：聚类问题（把相似的数据分成不同的组）、降维问题（把高维数据映射到低维空间）、异常检测问题（找出数据中与众不同的点）。
  
##  1-3.监督学习
  
  
- 监督学习是指给定一组训练样本（输入和输出），让计算机找出输入和输出之间的函数关系，并用这个函数来预测新的输入。
- 监督学习可以分为回归问题和分类问题：
  
    - 回归问题是指预测连续值的输出，例如房价、股票价格等。回归问题可以用直线或曲线来拟合数据，并用拟合得到的函数来预测新数据。
  
    - 分类问题是指预测离散值的输出，例如垃圾邮件、癌症类型等。分类问题可以用决策边界来划分数据，并用划分得到的区域来预测新数据。
  
##  1-4.无监督学习
  
  
- 无监督学习是指只给定一组输入数据，没有给定输出或标签，让计算机自己发现数据中隐藏的结构或模式。
- 无监督学习可以分为聚类问题和降维问题：
  
    - 聚类问题是指把相似的数据分成不同的组，例如市场细分、社交网络分析、图像压缩等。聚类问题可以用K-Means算法来解决。
  
    - 降维问题是指把高维数据映射到低维空间，以便于可视化或减少计算量，例如人脸识别、文本主题提取等。降维问题可以用主成分分析（PCA）算法来解决。
  