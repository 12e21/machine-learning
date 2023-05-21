import numpy as np
x = np.linspace(1, 100, 5) # 生成1到100的等差数列
y = np.linspace(1, 100, 5) # 生成1到100的等差数列
X, Y = np.meshgrid(x, y) # 生成网格点坐标矩阵
Z = np.vstack((X.ravel(), Y.ravel())).T # 把坐标矩阵转换成每一行是一对xy的形式


x=np.ones([3,2])
for i in range(3):
    print(i)