# pandas笔记

## 读写数据

pandas可以读写多种格式的数据，例如csv，excel，json，html等。常用的函数有：

- `pd.read_csv(filename)`：从csv文件中读取数据，返回一个DataFrame对象。
- `pd.to_csv(filename, df)`：将DataFrame对象保存到csv文件中。
- `pd.read_excel(filename)`：从excel文件中读取数据，返回一个DataFrame对象。
- `pd.to_excel(filename, df)`：将DataFrame对象保存到excel文件中。
- `pd.read_json(filename)`：从json文件中读取数据，返回一个DataFrame对象。
- `pd.to_json(filename, df)`：将DataFrame对象保存到json文件中。
- `pd.read_html(url)`：从网页中读取表格数据，返回一个列表。

## 创建数据

pandas可以创建两种主要的数据结构：Series和DataFrame。常用的函数有：

- `pd.Series(data, index)`：从列表，元组或字典创建一个一维的Series对象。
- `pd.DataFrame(data, index, columns)`：从列表，元组，字典或二维数组创建一个二维的DataFrame对象。
- `pd.date_range(start, end, freq)`：创建一个日期范围的Series对象。

## 操作数据

pandas提供了多种操作数据的方法，例如索引，切片，筛选，排序，分组，聚合等。常用的函数有：

- `df[index]`：根据索引访问或修改列或行。
- `df.loc[label]`：根据标签访问或修改列或行。
- `df.iloc[position]`：根据位置访问或修改列或行。
- `df[column]`：根据列名访问或修改列。
- `df[column][condition]`：根据条件筛选列或行。
- `df.sort_values(by, ascending)`：根据指定的列或行排序数据。
- `df.groupby(by)`：根据指定的列或行分组数据。
- `df.agg(func)`：对分组后的数据进行聚合操作。
- `df.merge(df2, on)`：根据指定的列或行合并两个DataFrame对象。
- `df.append(df2)`：将两个DataFrame对象沿着行方向拼接。
- `df.join(df2, on)`：根据指定的列或行连接两个DataFrame对象。

## 计算数据

pandas提供了多种计算数据的函数，例如数学运算，统计运算，时间序列运算等。常用的函数有：

- `df + df2`：对应元素相加，返回一个新的DataFrame对象。
- `df - df2`：对应元素相减，返回一个新的DataFrame对象。
- `df * df2`：对应元素相乘，返回一个新的DataFrame对象。
- `df / df2`：对应元素相除，返回一个新的DataFrame对象。
- `df ** n`：对应元素求n次幂，返回一个新的DataFrame对象。
- `df.sum(axis)`：沿着指定轴求和，返回一个Series对象或标量。
- `df.mean(axis)`：沿着指定轴求均值，返回一个Series对象或标量。
- `df.std(axis)`：沿着指定轴求标准差，返回一个Series对象或标量。
- `df.min(axis)`：沿着指定轴求最小值，返回一个Series对象或标量。
- `df.max(axis)`：沿着指定轴求最大值，返回一个Series对象或标量。
- `df.describe()`：对数据进行描述性统计分析，返回一个DataFrame对象。
- `df.corr()`：计算数据的相关系数矩阵，返回一个DataFrame对象。
- `df.cov()`：计算数据的协方差矩阵，返回一个DataFrame对象