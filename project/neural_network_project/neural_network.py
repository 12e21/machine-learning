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
params=[0.01*(np.random.rand(4,3)-0.5),0.01*(np.random.rand(4,5)-0.5),0.01*(np.random.rand(3,5)-0.5),0.01*(np.random.rand(2,4)-0.5)]
# 设置激活函数(sigmoid函数)
def sigmoid(x:np.ndarray):
    return 1/(1+np.exp(x))

# 设置layer,z,a
layer_count=4
layer=list(range(layer_count))
z=list(range(layer_count))
a=list(range(layer_count+1))
# 设置梯度
gradients=[np.zeros([4,3]),np.zeros([4,5]),np.zeros([3,5]),np.zeros([2,4])]


counter=0
# 迭代
while(counter<1):
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
    regular_rate=0.001
    # 误差计算公式
    loss=(-1./y.shape[1])*((y*np.log(hypo_x)+(1-y)*np.log(1-hypo_x)).flatten().sum()) + (regular_rate/(2*y.shape[0]))*sum([param.flatten().sum() for param in params])
    print(loss)

    # 反向传播
    errors=list(range(layer_count))
    # 最外层
    errors[layer_count-1]=a[4]-label

    for i in range(layer_count-1,-1,-1):
        errors[i-1]=np.dot(params[i][:,:-1].T,errors[i])*(a[i]*(1-a[i]))



    # 计算梯度
    for i in range(len(gradients)):
        gradients[i]=(1.0/y.shape[1])*np.dot(errors[i],layer[i].T)+regular_rate*params[i]

    
    # 梯度检测
    epsilon=0.001
    approx_params=params.copy()
    approx_gradients=[np.zeros([4,3]),np.zeros([4,5]),np.zeros([3,5]),np.zeros([2,4])]
    for i in range(len(approx_params)):
        for j in range(approx_params[i].shape[0]):
            for k in range(approx_params[i].shape[1]):
                
                approx_params[i][j,k]+=epsilon
                
                # 前向传播
                a[0]=feature.T
                for m in range(0,layer_count):
                    # 添加bias
                    layer[m]=np.concatenate((a[m],np.ones([1,a[m].shape[1]])),axis=0)
                    # 通过权重
                    z[m]=np.dot(approx_params[m],layer[m])
                    # 进入激活函数
                    a[m+1]=sigmoid(z[m])
                
                # 计算误差
                y=label
                hypo_x:np.ndarray = a[4]
                regular_rate=0.001
                # 误差计算公式
                approx_loss=(-1./y.shape[1])*((y*np.log(hypo_x)+(1-y)*np.log(1-hypo_x)).flatten().sum()) + (regular_rate/(2*y.shape[0]))*sum([param.flatten().sum() for param in approx_params])
                approx_gradients[i][j,k]=(approx_loss-loss)/epsilon
                
    
    # 更新参数
    learning_rate=0.3
    for i in range(len(params)):
        params[i]=params[i]-learning_rate*gradients[i]

    # 比较梯度
    
    print(gradients)
    print("!!!!!!!!!!!!!!!!!!")
    print(approx_gradients)
    

    counter+=1

    

