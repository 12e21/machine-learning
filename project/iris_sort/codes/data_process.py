import numpy as np
import pandas as pd
from machine_learn.utils import ensure_path_sep

# 数据处理
# 读入数据

data=pd.read_csv(ensure_path_sep("../data/Iris.csv"))

# 分离特征矩阵和标签向量
label=data.loc[:,"Species"]
feature=data.loc[:,:"PetalWidthCm"]


# 对标签进行onehot编码
label=pd.get_dummies(label)

np.savetxt(ensure_path_sep("../data/feature.txt"),feature.values)

np.savetxt(ensure_path_sep("../data/label.txt"),label.values)

print(label)
