import numpy as np
import matplotlib.pyplot as plt
'''
本程序为一个线性回归实例
'''

# 正向预测函数
def hypothetical(theta_0:float,theta_1:float,x:np.ndarray)->np.ndarray:
    y=theta_0+theta_1*x
    return y
# 损失函数
def cal_loss(theta_0:float,theta_1:float,train_x:np.ndarray,train_y:np.ndarray)->float:
    #预测的x值
    predict_x=hypothetical(theta_0,theta_1,train_x)
    #方差
    square_diff=np.sum(np.power(predict_x-train_y,2))
    return square_diff/(2*(train_x.size))

# theta_0(损失)梯度计算函数
def cal_theta_0_gradient(theta_0:float,theta_1:float,train_x:np.ndarray,train_y:np.ndarray)->float:
    predict_x=hypothetical(theta_0,theta_1,train_x)
    return np.sum(predict_x-train_y)/train_x.size
# theta_1(损失)梯度计算函数
def cal_theta_1_gradient(theta_0:float,theta_1:float,train_x:np.ndarray,train_y:np.ndarray)->float:
    predict_x=hypothetical(theta_0,theta_1,train_x)
    #中间计算
    middle=np.dot((predict_x-train_y),train_x)
    return middle/train_x.size

# 更新参数
def update_param(theta:float,gradient:float,learning_rate:float):
    return theta-learning_rate*gradient


if __name__=="__main__":
    #模拟一个训练集
    train_x=np.array([1,2,3,4,5,6,7,8,9,10])
    train_y=np.array([1,4,6,9,10,9,16,16,18,19])


    #设置初始theta值
    theta_0=0
    theta_1=0

    #设置迭代次数计数器,误差,学习率
    iter_counter=0
    loss=3
    pre_loss=1
    learning_rate=0.05

    #开始迭代直至到达最大循环次数或收敛
    while not(iter_counter==1000 or abs(loss-pre_loss) < 0.0001):
        # 计算梯度
        theta_0_gradient=cal_theta_0_gradient(theta_0,theta_1,train_x,train_y)
        theta_1_gradient=cal_theta_1_gradient(theta_0,theta_1,train_x,train_y)
        

        # 更新参数
        theta_0=update_param(theta_0,theta_0_gradient,learning_rate)
        theta_1=update_param(theta_1,theta_1_gradient,learning_rate)

        # 画出拟合曲线
        plt.plot(train_x,hypothetical(theta_0,theta_1,train_x),color='red')
        plt.scatter(train_x,train_y)
        plt.pause(0.1)
        plt.clf()

        #计算误差
        pre_loss=loss
        loss=cal_loss(theta_0,theta_1,train_x,train_y)
        print(loss)

        iter_counter+=1

    print(f"最终拟合结果 theta0:{theta_0} ,theta1: {theta_1}")







