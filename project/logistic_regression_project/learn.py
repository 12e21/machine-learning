import numpy as np
class BaseLearning:
    def __init__(self,featureMmatrix:np.ndarray,labelMatrix:np.ndarray) -> np.ndarray:
        '''
        构造函数
        featureMmatrix: 特征矩阵
        labelMatrix : 标签向量
        返回: 预测结果
        '''
        self.featureMmatrix=featureMmatrix
        self.labelMatrix=labelMatrix


    def _hypothetical(self):
        pass

    def _cal_loss(self):
        pass

    def cal_gradient(self):
        pass

    def iterate(self):
        pass

# TODO 完善基类
