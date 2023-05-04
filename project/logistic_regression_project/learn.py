import numpy as np
import utils
# template learn class
class BaseLearning:
    def __init__(self,featureMmatrix:np.ndarray,labelVector:np.ndarray,paramsVector:np.ndarray):
        '''
        构造函数
        featureMmatrix: 特征矩阵
        labelVector : 标签向量
        paramsVector : 参数向量
        返回: 预测结果
        '''
        self._featureMmatrix=featureMmatrix
        self._labelVector=labelVector
        self._paramsVector=paramsVector


    def _hypothetical(self):
        pass

    def _cal_loss(self):
        pass

    def _cal_gradient(self):
        pass

    def iterate(self,learningRate:float,maxIterCount:int):
        pass

    def save_model(self):
        np.savetxt("modelParams",self._paramsVector)


class Linear_regression(BaseLearning):
    def __init__(self, featureMmatrix: np.ndarray, labelVector: np.ndarray, paramsVector: np.ndarray):
        super().__init__(featureMmatrix, labelVector, paramsVector)

    def _hypothetical(self):
        return super()._hypothetical()

# 逻辑回归子类    
class Logistic_regression(BaseLearning):
    def __init__(self, featureMmatrix: np.ndarray, labelVector: np.ndarray, paramsVector: np.ndarray):
        super().__init__(featureMmatrix, labelVector, paramsVector)

    def _hypothetical(self):
        y=-(np.dot(self._featureMmatrix,self._paramsVector.T))
        y=1/(1+np.exp(y))
        return y
    
    def _cal_loss(self):
        predict=self._hypothetical()
        m=self._featureMmatrix.shape[0]
        left=self._labelVector*np.log(predict)
        right=(1-self._labelVector)*np.log(1-predict)
        result=(-1.0/m)*np.sum(left+right)
        return result
    
    def _cal_gradient(self):
        predict=self._hypothetical()
        diff=predict-self._labelVector
        middle=np.dot(self._featureMmatrix.T,diff)
        return middle/self._featureMmatrix.shape[0]
    
    def iterate(self, learningRate: float, maxIterCount: int):
        #设置迭代次数计数器,误差,学习率
        iter_counter=0
        loss=3
        pre_loss=1
        learning_rate=learningRate

        #开始迭代直至到达最大循环次数或收敛
        while not(iter_counter==maxIterCount or abs(loss-pre_loss) < 0.0000003):
            # 计算梯度
            gradients=self._cal_gradient()

            # 更新梯度
            self._paramsVector=self._paramsVector-learning_rate*gradients

            # 计算误差
            pre_loss=loss
            loss=self._cal_loss()
            
            # 打印误差
            print(loss)

    def val(self):
        print(self._hypothetical())


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

    l1 = Logistic_regression(x,y,theta)
    l1.iterate(0.0003,3000000)
    l1.save_model()
    l1.val()