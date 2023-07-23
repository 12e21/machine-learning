'''
import numpy as np
x = np.linspace(1, 100, 5) # 生成1到100的等差数列
y = np.linspace(1, 100, 5) # 生成1到100的等差数列
X, Y = np.meshgrid(x, y) # 生成网格点坐标矩阵
Z = np.vstack((X.ravel(), Y.ravel())).T # 把坐标矩阵转换成每一行是一对xy的形式


x=np.ones([3,2])
for i in range(3):
    print(i)s
'''
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

row_matrix = arr[0:1, :]  # 使用切片操作选择第一行
print(row_matrix)
column_matrix = arr[:, 0:1]  # 使用切片操作选择第一列
print(column_matrix)