import numpy as np

# 设置特征矩阵和标签(OR关系)
feature=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])
label=np.array([[0,0,0,1]]).T

# 设置参数矩阵集
params=[np.zeros([4,3]),np.zeros([4,5]),np.zeros([3,5]),np.zeros([1,4])]

# 设置激活函数(sigmoid函数)
def sigmoid(x:np.ndarray):
    return 1/(1+np.exp(x))

# 添加bias
layer0=np.concatenate((feature.T,np.ones([1,feature.T.shape[1]])),axis=0)
# 通过权重
z_0=np.dot(params[0],layer0)
# 进入激活函数
a_1=sigmoid(z_0)

# 添加bias
layer1=np.concatenate((a_1,np.ones([1,a_1.shape[1]])),axis=0)
# 通过权重
z_1=np.dot(params[1],layer1)
# 进入激活函数
a_2=sigmoid(z_1)

# 添加bias
layer2=np.concatenate((a_2,np.ones([1,a_2.shape[1]])),axis=0)
# 通过权重
z_2=np.dot(params[2],layer2)
# 进入激活函数
a_3=sigmoid(z_2)

# 添加bias
layer3=np.concatenate((a_3,np.ones([1,a_3.shape[1]])),axis=0)
# 通过权重
z_3=np.dot(params[3],layer3)
# 进入激活函数
a_4=sigmoid(z_3)

print(a_4)
