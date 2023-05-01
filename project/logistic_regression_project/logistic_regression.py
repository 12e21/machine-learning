import numpy as np
import matplotlib as plt
import utils
'''
本程序为
'''


def hypothetical(theta:np.ndarray,x:np.ndarray)->np.ndarray:
    '''
    预测函数
    theta: 参数向量
    x: 特征矩阵
    返回: 预测结果向量
    '''
    y=-(np.dot(x,theta.T))
    y=1/(1+np.exp(y))
    return y


def cal_loss(theta:np.ndarray,x:np.ndarray,y:np.ndarray):
    '''
    损失函数
    theta: 参数向量
    x: 特征矩阵
    y: 标签矩阵
    返回: loss(标量)
    '''
    predict=hypothetical(theta,x)
    m=x.shape[0]
    left=y*np.log(predict)
    right=(1-y)*np.log(1-predict)
    result=(-1.0/m)*np.sum(left+right)
    return result


def cal_gradient(theta:np.ndarray,x:np.ndarray,y:np.ndarray)->np.ndarray:
    '''
    梯度函数
    theta: 参数向量
    x: 特征矩阵
    y: 标签矩阵
    返回: 梯度向量
    '''
    predict=hypothetical(theta,x)
    diff=predict-y
    middle=np.dot(x.T,diff)
    return middle/x.shape[0]



if __name__ == "__main__":
    x=np.array([[1,11,0.91],
                [1,12,0.88],
                [1,14,0.72],
                [1,20,0.63],
                [1,24,0.51],
                [1,28,0.42],
                [1,29,0.33]])
    
    x[:,1:]=utils.normalize_matrix(x[:,1:])
    y=np.array([1,1,1,0,0,0,0])
    theta=np.array([0.1,0.2,0.4])

    #设置迭代次数计数器,误差,学习率
    iter_counter=0
    loss=3
    pre_loss=1
    learning_rate=0.0001

    #开始迭代直至到达最大循环次数或收敛
    while not(iter_counter==1000000 or abs(loss-pre_loss) < 0.00000003):
        # 计算梯度
        gradients=cal_gradient(theta,x,y)

        # 更新梯度
        theta=theta-learning_rate*gradients

        # 计算误差
        pre_loss=loss
        loss=cal_loss(theta,x,y)
        
        # 打印误差
        print(loss)

    # 打印最后参数
    print(theta)
    # 预测结果
    result=hypothetical(theta,x)
    result[np.where(result>0.50000)]=1
    result[np.where(result<0.50000)]=0
    print(result)