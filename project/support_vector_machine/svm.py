import numpy as np
import utils
class Svm:
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
        self._variRate=0.01
        self.learningRate=0.003


    def _hypothetical(self,currentFeature)->float:
        '''
        :param currentFeature: 当前进行计算的特征
        :return: 特征与参数的内积
        '''
        return float(np.dot(currentFeature,self._paramsVector))

    def cost1(self,x:float)->float:
        if x>=1 :
            return 0.
        else:
            return -0.75*(x-1.)

    def cost2(self,x:float)->float:
        if x>=-1:
            return 0.75*(x+1.)
        else:
            return 0.


    def _cal_loss(self)->float:
        sumLoss=0.
        for i in range(self._featureMmatrix.shape[0]):
            hypo=self._hypothetical(currentFeature=self._featureMmatrix[i,:])
            sumLoss+=(self._variRate/self._featureMmatrix.shape[0])*(float(self._labelVector[i,:])*self.cost1(hypo)+float(1.-self._labelVector[i,:])*self.cost2(hypo))

        sumLoss+=+ (1./2.)*(np.power(self._paramsVector.flatten(),2).sum())
        return sumLoss

    def gradient1(self,x:float)->float:
        if x >= 1.:
            return 0
        else:
            return -0.75

    def gradient2(self,x:float)->float:
        if x>=-1:
            return 0.75
        else:
            return 0


    def _cal_gradient(self)->np.ndarray:
        sumGradient=np.zeros(self._paramsVector.shape)
        for i in range(self._featureMmatrix.shape[0]):
            hypo = self._hypothetical(currentFeature=self._featureMmatrix[i, :])
            sumGradient+= (self._variRate / self._featureMmatrix.shape[0])*(float(self._labelVector[i, :]) * self.cost1(hypo)*self._featureMmatrix[i,:].reshape(self._paramsVector.shape) +
                           float(1. - self._labelVector[i, :]) * self.cost2(hypo)*self._featureMmatrix[i,:].reshape(self._paramsVector.shape))

        sumGradient+=self._paramsVector
        return sumGradient
    def iterate(self):
        for i in range(5000000):
            loss=self._cal_loss()
            self._paramsVector-=self.learningRate*self._cal_gradient()
            print(loss)


if __name__ == "__main__":
    x = np.array([[1, 11, 0.91],
                  [1, 12, 0.88],
                  [1, 14, 0.72],
                  [1, 20, 0.63],
                  [1, 24, 0.51],
                  [1, 28, 0.42],
                  [1, 29, 0.33]])

    x[:, 1:] = utils.normalize_matrix(x[:, 1:])
    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0],
                  [0],
                  [0]])
    theta = np.array([[0.1],
                     [0.2],
                     [0.4]])
    mySvm=Svm(featureMmatrix=x,labelVector=y,paramsVector=theta)
    mySvm.iterate()



