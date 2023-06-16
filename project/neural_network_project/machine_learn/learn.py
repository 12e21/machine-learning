import numpy as np

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

    def get_params(self)->np.ndarray:
        return self._paramsVector

    def save_model(self):
        np.savetxt("modelParams",self._paramsVector)

# 线性回归子类
class Linear_regression(BaseLearning):
    def __init__(self, featureMmatrix: np.ndarray, labelVector: np.ndarray, paramsVector: np.ndarray):
        super().__init__(featureMmatrix, labelVector, paramsVector)

    def _hypothetical(self):
        return np.dot(self._featureMmatrix,self._paramsVector.T)

    def _cal_loss(self):
        predict=self._hypothetical()
        square_diff=(predict-self._labelVector)**2
        return np.sum(square_diff)/(2*self._featureMmatrix.shape[0])

    def _cal_gradient(self):
        predict=self._hypothetical()
        diff=predict-self._labelVector
        middle=np.dot(self._featureMmatrix.T,diff)
        return middle/self._featureMmatrix.shape[0]   
    
    def _cal_regular_item(self,regularRate:float):
        return (regularRate/self._featureMmatrix.shape[0])*self._paramsVector
    
    def iterate(self, learningRate: float, maxIterCount: int,regularRate:float):
        #设置迭代次数计数器,误差,学习率
        iter_counter=0
        loss=3
        pre_loss=1
        learning_rate=learningRate

        #开始迭代直至到达最大循环次数或收敛
        while not(iter_counter==maxIterCount or abs(loss-pre_loss) < 0.0000003):
            # 计算梯度
            gradients=self._cal_gradient()
            
            # 计算正则项
            regularItem=self._cal_regular_item(regularRate)

            # 更新梯度
            self._paramsVector=self._paramsVector-learning_rate*(gradients+regularItem)

            # 计算误差
            pre_loss=loss
            loss=self._cal_loss()
            
            # 打印误差
            print(loss)

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
    
    def _cal_regular_item(self,regularRate:float):
        '''
        计算正则项
        regularRate : 正则率
        '''
        return (regularRate/self._featureMmatrix.shape[0])*self._paramsVector

    def iterate(self, learningRate: float, maxIterCount: int,regularRate:float):
        #设置迭代次数计数器,误差,学习率
        iter_counter=0
        loss=3
        pre_loss=1
        learning_rate=learningRate

        #开始迭代直至到达最大循环次数或收敛
        while not(iter_counter==maxIterCount or abs(loss-pre_loss) < 0.0000003):
            # 计算梯度
            gradients=self._cal_gradient()
            
            # 计算正则项
            regularItem=self._cal_regular_item(regularRate)

            # 更新梯度
            self._paramsVector=self._paramsVector-learning_rate*(gradients+regularItem)

            # 计算误差
            pre_loss=loss
            loss=self._cal_loss()
            
            # 打印误差
            print(loss)

    def val(self):
        print(self._hypothetical())

if __name__ == "__main__":
    # 生成测试linear_regression的数据
    featureMmatrix=np.array([[1,2,3],[1,3,4],[1,4,5],[1,5,6]])
    labelVector=np.array([1,2,3,4])
    paramsVector=np.array([1,1,1])
    
    # 测试linear_regression
    linear_regression=Linear_regression(featureMmatrix,labelVector,paramsVector)
    linear_regression.iterate(0.01,1000,0.1)
    print(linear_regression.get_params())

    
