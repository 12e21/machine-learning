import numpy as np


# 设置特征和标签(规律为后面两个数取并再与第一个数取交)
feature=np.array([
    [0,0,0,1,1,1,0,1],
    [0,0,1,0,1,0,1,1],
    [0,1,0,0,0,1,1,1],
])

label=np.array([
    [0,0,0,0,1,1,0,1],
    [1,1,1,1,0,0,1,0]
    ])

# 设置神经元层数
layer_count=4
# 设置每层神经元的神经元数量
neural_count_of_layers=[3,4,3,2]


# 激活项
activate_item=[np.zeros([i,1]) for i in neural_count_of_layers]
# 中间项
z_item=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]
# 权重
weights=[np.random.rand(neural_count_of_layers[i+1],neural_count_of_layers[i]) for i in range(layer_count-1)]
# 偏差权重
bias_weights=[np.random.rand(i,1) for i in neural_count_of_layers[1:]]
# 合成参数
thetas=[np.concatenate((weights[i],bias_weights[i]),1) for i in range(layer_count-1)]


# epsilon项
epsilons=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]
# 权重梯度项
deltas=[np.zeros([neural_count_of_layers[i+1],neural_count_of_layers[i]]) for i in range(layer_count-1)]
# 偏差梯度项
bias_deltas=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]

# 误差
loss=[0,0]
learning_rate=0.3


# 设置激活函数(sigmoid函数)
def sigmoid(x:np.ndarray):
    return 1/(1+np.exp(-x))

# 前向传播(一组数据)
def forward_propagation():
    for i in range(layer_count-1):
        # a添加bias
        a_bias=np.concatenate((activate_item[i],[[1]]),0)
        # 乘参数 
        z_item[i]=np.dot(thetas[i],a_bias)
        # 激活
        activate_item[i+1]=sigmoid(z_item[i])


# 计算误差(一组数据)
def cal_loss(current_label:np.ndarray):
    loss[0]+=(current_label*np.log(activate_item[-1])+(1-current_label)*np.log(1-activate_item[-1])).flatten().sum()
    
# 反向传播
def back_propagation(current_label:np.ndarray):
    # 最后一个epsilon
    epsilons[-1]=activate_item[-1]-current_label
    for i in range(layer_count-3,-1,-1):
        epsilons[i]=np.dot(weights[i+1].transpose(),epsilons[i+1])*(activate_item[i+1]*(1-activate_item[i+1]))
        
# 计算梯度
def cal_delta():
    for i in range(layer_count-2,-1,-1):
        deltas[i]+=np.dot(epsilons[i],activate_item[i].transpose())
        bias_deltas[i]+=epsilons[i]

# 对一组数据的完整过程
def cal_all(current_feature:np.ndarray,current_label:np.ndarray):
    activate_item[0]=current_feature
    forward_propagation()
    cal_loss(current_label)
    back_propagation(current_label)
    cal_delta()

# 梯度下降
def gradient_decrease():
    for i in range(layer_count-1):
        weights[i]-=(1./feature.shape[1])*learning_rate*deltas[i]
        bias_weights[i]-=(1./feature.shape[1])*learning_rate*bias_deltas[i]
    
    global thetas
    thetas=[np.concatenate((weights[i],bias_weights[i]),1) for i in range(layer_count-1)]

# 一次完整迭代
def one_iterate():
    # 对每组数据前向传播和反向传播
    for i in range(label.shape[1]):
        cal_all(feature[:,i].reshape([feature[:,i].size,1]),label[:,i].reshape([label[:,i].size,1]))
        #print(str(i)+":"+str(activate_item[-1]))
    gradient_decrease()

    print(loss[0]*(-1./feature.shape[1]))
    loss[0]=0
    # 权重梯度项
    global deltas
    deltas=[np.zeros([neural_count_of_layers[i+1],neural_count_of_layers[i]]) for i in range(layer_count-1)]
    # 偏差梯度项
    global bias_deltas
    bias_deltas=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]


for i in range(10000000000000):
    one_iterate()



'''
forward_propagation()
# 此处记得reshape
cal_loss(label[:,0].reshape([label[:,0].size,1]))
print(loss[0])
cal_delta()
'''


