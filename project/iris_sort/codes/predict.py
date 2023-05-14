import numpy as np
from machine_learn import utils
from machine_learn import learn

# 获取数据
feature=np.loadtxt("project/iris_sort/data/feature.txt")

params_1=np.loadtxt("project/iris_sort/data/model/params_1.txt")
params_2=np.loadtxt("project/iris_sort/data/model/params_2.txt")
params_3=np.loadtxt("project/iris_sort/data/model/params_3.txt")

label=np.ones(feature.shape[0])

# 将每个特征扩展到3次
feature=utils.reflect_to_high_dim(feature,4,3)
feature=utils.reflect_to_high_dim(feature,3,3)
feature=utils.reflect_to_high_dim(feature,2,3)
feature=utils.reflect_to_high_dim(feature,1,3)
feature=utils.reflect_to_high_dim(feature,1,3)

# 将feature矩阵归一化
feature=utils.normalize_matrix(feature)

# 添加单位特征
feature=np.concatenate((np.ones([feature.shape[0],1]),feature),1)

# 创建子模型
sub_model1=learn.Logistic_regression(feature,label,params_1)
sub_model2=learn.Logistic_regression(feature,label,params_2)
sub_model3=learn.Logistic_regression(feature,label,params_3)

# 模型预测


result=np.concatenate((np.expand_dims(sub_model1._hypothetical(),axis=0).transpose()
                       ,np.expand_dims(sub_model2._hypothetical(),axis=0).transpose()
                       ,np.expand_dims(sub_model3._hypothetical(),axis=0).transpose()),axis=1)

# 选取最大概率的类别作为预测的类别
result=np.argmax(result,axis=1)
print(result)