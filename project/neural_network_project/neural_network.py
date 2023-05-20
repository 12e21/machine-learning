import numpy as np

# 设置特征矩阵和标签(OR关系)
feature=np.array([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]])
label=np.array([[1.,1.,1.,0.],
                [0.,0.,0.,1.]])

# 设置参数矩阵集
params=[np.ones([4,3]),np.ones([4,5]),np.ones([3,5]),np.ones([2,4])]
# 设置激活函数(sigmoid函数)
def sigmoid(x:np.ndarray):
    return 1/(1+np.exp(x))

# 设置layer,z,a
layer_count=4
layer=list(range(layer_count))
z=list(range(layer_count))
a=list(range(layer_count+1))

# 前向传播
a[0]=feature.T
for i in range(0,layer_count):
    # 添加bias
    layer[i]=np.concatenate((a[i],np.ones([1,a[i].shape[1]])),axis=0)
    # 通过权重
    z[i]=np.dot(params[i],layer[i])
    # 进入激活函数
    a[i+1]=sigmoid(z[i])


# 计算误差
y=label
hypo_x:np.ndarray = a[4]
regular_rate=0.003
# 误差计算公式
loss=(-1./y.shape[1])*((y*np.log(hypo_x)+(1-y)*np.log(1-hypo_x)).flatten().sum()) + (regular_rate/(2*y.shape[0]))*sum([param.flatten().sum() for param in params])



# 反向传播
errors=list(range(layer_count))
# 最外层
errors[layer_count-1]=a[4]-label

for i in range(layer_count-1,-1,-1):
    errors[i-1]=np.dot(params[i][:,:-1].T,errors[i])*(a[i]*(1-a[i]))


# 设置梯度
gradients=[np.zeros([4,2]),np.zeros([4,4]),np.zeros([3,4]),np.zeros([2,3])]
# 计算梯度
for i in range(len(gradients)):
    gradients[i]=(1.0/y.shape[1])*gradients[i]+np.dot(errors[i],a[i].T)+regular_rate*params[i][:,:-1]

# 更新参数
learning_rate=0.002
for i in range(len(params)):
    params[i][:,:-1]=params[i][:,:-1]-learning_rate*gradients[i]

print(params)
    

