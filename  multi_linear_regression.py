import numpy as np
import matplotlib.pyplot as plt

'''
本程序为一个多元线性回归实例(梯度下降法)
'''

def hypothetical(theta:np.ndarray,x:np.ndarray)->np.ndarray:
    '''
    预测函数
    
    theta: 参数向量
    x: 特征矩阵
    y: 标签向量
    '''
    y=np.dot(x,theta.T)
    return y


'''
损失函数(输入参数向量,特征矩阵,标签矩阵,返回loss)
'''
def cal_loss(theta:np.ndarray,x:np.ndarray,y:np.ndarray):
    predict=hypothetical(theta,x)
    square_diff=(predict-y)**2
    return np.sum(square_diff)/(2*x.shape[0])


'''
梯度函数(输入参数向量,特征矩阵,标签矩阵,返回梯度向量)
'''
def cal_gradient(theta:np.ndarray,x:np.ndarray,y:np.ndarray)->np.ndarray:
    predict=hypothetical(theta,x)
    diff=predict-y
    middle=np.dot(x.T,diff)
    return middle/x.shape[0]
    



if __name__ == "__main__":
    # 测试集
    theta=np.array([0,0,0])

    x=np.array([[1,20,0.3],
                [1,22,0.1],
                [1,18,0.4],
                [1,17,0.6],
                [1,19,0.5],
                [1,21,0.2],
                [1,23,0.1]])
    
    y=np.array([47,41.3,39.2,38.5,40.2,42.1,43.2])

    #设置迭代次数计数器,误差,学习率
    iter_counter=0
    loss=3
    pre_loss=1
    learning_rate=0.0000001

    #开始迭代直至到达最大循环次数或收敛
    while not(iter_counter==1000000 or abs(loss-pre_loss) < 0.0000001):
        # 计算梯度
        gradients=cal_gradient(theta,x,y)

        # 更新梯度
        theta=theta-learning_rate*gradients

        # 计算误差
        pre_loss=loss
        loss=cal_loss(theta,x,y)
        
        # 打印误差
        print(loss)

    print(theta)
    
