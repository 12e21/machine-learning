import numpy as np
from machine_learn import utils
from machine_learn import learn

# 获取数据
feature=np.loadtxt("project/iris_sort/data/feature.txt")
label=np.loadtxt("project/iris_sort/data/label.txt")


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

# 分离label向量
label1=label[:,0]
label2=label[:,1]
label3=label[:,2]

# 创建参数
params_1=np.ones([1,feature.shape[1]]).flatten()
params_2=np.ones([1,feature.shape[1]]).flatten()
params_3=np.ones([1,feature.shape[1]]).flatten()

# 创建子模型
sub_model1=learn.Logistic_regression(feature,label1,params_1)
sub_model2=learn.Logistic_regression(feature,label2,params_2)
sub_model3=learn.Logistic_regression(feature,label3,params_3)

# 模型迭代
sub_model1.iterate(0.0003,10000,1.0)
print("sub model 1 finished !")
sub_model2.iterate(0.0003,10000,1.0)
print("sub model 2 finished !")
sub_model3.iterate(0.0003,10000,1.0)
print("sub model 3 finished !")

# 获取参数
params_1=sub_model1.get_params()
params_2=sub_model2.get_params()
params_3=sub_model3.get_params()

# 储存参数
np.savetxt("project/iris_sort/data/model/params_1.txt",params_1)
np.savetxt("project/iris_sort/data/model/params_2.txt",params_2)
np.savetxt("project/iris_sort/data/model/params_3.txt",params_3)

print("all finished !")