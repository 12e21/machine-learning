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

# 设置layer,z,a
layer_count=4
layer=list(range(layer_count))
z=list(range(layer_count))
a=list(range(layer_count+1))

a[0]=feature.T

# 添加bias
layer[0]=np.concatenate((a[0],np.ones([1,a[0].shape[1]])),axis=0)
# 通过权重
z[0]=np.dot(params[0],layer[0])
# 进入激活函数
a[1]=sigmoid(z[0])

# 添加bias
layer[1]=np.concatenate((a[1],np.ones([1,a[1].shape[1]])),axis=0)
# 通过权重
z[1]=np.dot(params[1],layer[1])
# 进入激活函数
a[2]=sigmoid(z[1])

# 添加bias
layer[2]=np.concatenate((a[2],np.ones([1,a[2].shape[1]])),axis=0)
# 通过权重
z[2]=np.dot(params[2],layer[2])
# 进入激活函数
a[3]=sigmoid(z[2])

# 添加bias
layer[3]=np.concatenate((a[3],np.ones([1,a[3].shape[1]])),axis=0)
# 通过权重
z[3]=np.dot(params[3],layer[3])
# 进入激活函数
a[4]=sigmoid(z[3])

print(a[4])
