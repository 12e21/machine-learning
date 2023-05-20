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

for i in range(0,layer_count):
    # 添加bias
    layer[i]=np.concatenate((a[i],np.ones([1,a[i].shape[1]])),axis=0)
    # 通过权重
    z[i]=np.dot(params[i],layer[i])
    # 进入激活函数
    a[i+1]=sigmoid(z[i])

print(a[4])
