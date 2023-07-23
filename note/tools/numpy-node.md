# numpy笔记

## 读写数据

numpy可以读写多种格式的数据，例如csv，txt，hdf5等。常用的函数有：

- `np.loadtxt(filename)`：从文本文件中读取数据，返回一个数组。
- `np.savetxt(filename, arr)`：将数组保存到文本文件中。
- `np.genfromtxt(filename)`：从文本文件中读取数据，可以处理缺失值，返回一个结构化数组。
- `np.load(filename)`：从二进制文件中读取数据，返回一个数组或字典。
- `np.save(filename, arr)`：将数组保存到二进制文件中。
- `np.savez(filename, **kwargs)`：将多个数组保存到一个压缩的二进制文件中。

## 创建数据

numpy可以创建各种类型和形状的数组，常用的函数有：

- `np.array(obj)`：从列表，元组或其他序列类型创建一个数组。
- `np.arange(start, stop, step)`：创建一个等差数列的数组。
- `np.linspace(start, stop, num)`：创建一个等分数列的数组。
- `np.zeros(shape)`：创建一个全零的数组。
- `np.ones(shape)`：创建一个全一的数组。
- `np.eye(N)`：创建一个N阶单位矩阵。
- `np.random.rand(shape)`：创建一个服从均匀分布的随机数组。
- `np.random.randn(shape)`：创建一个服从标准正态分布的随机数组。
- `np.random.randint(low, high, shape)`：创建一个服从离散均匀分布的随机整数数组。

## 操作数据

numpy提供了多种操作数组的方法，例如索引，切片，变形，合并，分割等。常用的函数有：

- `arr[index]`：根据索引访问或修改数组元素。
- `arr[start:stop:step]`：根据切片访问或修改数组子集。
- `arr.reshape(shape)`：改变数组的形状，返回一个新的数组。
- `arr.flatten()`：将数组展平为一维，返回一个新的数组。
- `arr.transpose()`：转置数组，返回一个新的数组。
- `np.concatenate((arr1, arr2), axis)`：沿着指定轴合并两个或多个数组，返回一个新的数组。
- `np.split(arr, indices_or_sections, axis)`：沿着指定轴分割一个数组，返回一个列表。
-  注意 `matrix[0,:]`的切片是一个tuple,而`matrix[0:1,:]的切片是矩阵`
## 计算数据

numpy提供了多种计算数组的函数，例如数学运算，统计运算，线性代数运算等。常用的函数有：

- `arr + arr2`：对应元素相加，返回一个新的数组。
- `arr - arr2`：对应元素相减，返回一个新的数组。
- `arr * arr2`：对应元素相乘，返回一个新的数组。
- `arr / arr2`：对应元素相除，返回一个新的数组。
- `arr ** n`：对应元素求n次幂，返回一个新的数组。
- `np.dot(arr, arr2)`：计算两个数组的点积，返回一个标量或数组。
- `np.sum(arr, axis)`：沿着指定轴求和，返回一个标量或数组。
- `np.mean(arr, axis)`：沿着指定轴求均值，返回一个标量或数组。
- `np.std(arr, axis)`：沿着指定轴求标准差，返回一个标量或数组。
- `np.min(arr, axis)`：沿着指定轴求最小值，返回一个标量或数组。
- `np.max(arr, axis)`：沿着指定轴求最大值，返回一个标量或数组。
- `np.linalg.inv(arr)`：计算数组的逆矩阵，返回一个新的数组。
- `np.linalg.det(arr)`：计算数组的行列式，返回一个标量。
- `np.linalg.eig(arr)`：计算数组的特征值和特征向量，返回一个元组。
- `np.sin(arr)`：对应元素求正弦值，返回一个新的数组。
- `np.cos(arr)`：对应元素求余弦值，返回一个新的数组。
- `np.tan(arr)`：对应元素求正切值，返回一个新的数组。
- `np.exp(arr)`：对应元素求自然指数，返回一个新的数组。
- `np.log(arr)`：对应元素求自然对数，返回一个新的数组。