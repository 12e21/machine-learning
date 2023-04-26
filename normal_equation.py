import numpy as np
'''
解一个正规方程
'''
def cal_normal_equation(x:np.ndarray,y:np.ndarray)->np.ndarray:
    '''
    x: 特征矩阵
    y: 标签向量
    return: 参数向量
    '''
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.swapaxes(0,1),x)),x.swapaxes(0,1)),y)
    
    
if __name__ == "__main__":
    x=np.array([[1,20,0.3],
                [1,22,0.1],
                [1,18,0.4],
                [1,17,0.6],
                [1,19,0.5],
                [1,21,0.2],
                [1,23,0.1]])
    
    y=np.array([47,41.3,39.2,38.5,40.2,42.1,43.2])

    print(cal_normal_equation(x,y))
    
    
