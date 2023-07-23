import utils
import numpy as np
import matplotlib.pyplot as plt

'''
本程序为一个多项式线性回归实例(梯度下降法)
'''

'''
预测函数(输入参数向量和特征矩阵,返回结果的向量)
'''
def hypothetical(theta:np.ndarray,x:np.ndarray)->np.ndarray:
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
    # 设置测试集
    theta=np.array([0,0,0,0,0,0])
    x=np.array([[1.,1.],
                [1.,2.],
                [1.,3.],
                [1.,4.],
                [1.,5.],
                [1.,6.],
                [1.,7.],
                [1.,8.],
                [1.,9.],
                [1.,10.],
                [1.,11.],
                [1.,12.],
                [1.,13.]])
    y=np.array([12.3,12.4,10.45,8.76,3.24,0.72,1.67,2.33,3.11,4.89,5.45,7.80,11.2])
    # 拓展x维度
    x=utils.reflect_to_high_dim(x,1,5)
    # 归一化x
    x[:,1:]=utils.normalize_matrix(x[:,1:])



    #设置迭代次数计数器,误差,学习率
    iter_counter=0
    loss=3
    pre_loss=1
    learning_rate=1.1

    #开始迭代直至到达最大循环次数或收敛
    while not(iter_counter==1000000 or abs(loss-pre_loss) < 0.0000000001):
        # 计算梯度
        gradients=cal_gradient(theta,x,y)

        # 更新梯度
        theta=theta-learning_rate*gradients

        # 计算误差
        pre_loss=loss
        loss=cal_loss(theta,x,y)

        # 画出图像
        plt.scatter(x[:,1],y)
        plt.plot(x[:,1],hypothetical(theta,x),color='red')
        plt.savefig('F:\code\zhihu_blog\machine_learning\pic'+"\\"+str(iter_counter)+".jpg")
        plt.show()
        plt.clf()

        iter_counter+=1
        # 打印误差
        print(loss)
    
    print(theta)