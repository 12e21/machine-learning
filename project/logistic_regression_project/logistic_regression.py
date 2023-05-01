import numpy as np
import matplotlib as plt
import utils



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



if __name__ == "__main__":
    x=np.array([[1,20,0.3],
                [1,22,0.1],
                [1,18,0.4],
                [1,17,0.6],
                [1,19,0.5],
                [1,21,0.2],
                [1,23,0.1]])
    
    y=np.array([1,1,1,0,1,0,1])
    theta=np.array([0.1,0.2,0.4])
    print(hypothetical(theta,x))
    print(cal_loss(theta,x,y))