import numpy as np
import os
'''
本文件存放一些工具函数
'''

'''
保持windows和linux上的路径兼容
'''
def ensure_path_sep(path:str) -> str:
    if "/" in path:
        path = os.sep.join(path.split("/"))
    if "\\\\" in path:
        path = os.sep.join(path.split("\\\\"))
    return path


'''
标准化一个正则向量
'''
def standardize_vector(x:np.ndarray)->np.ndarray:
    mean=np.mean(x)
    std=np.std(x)
    result=(x-mean)/std
    return result

'''
标准化一个特征矩阵中的所有特征向量
'''
def standardize_matrix(x:np.ndarray)->np.ndarray:
    for fearture_index in range(x.shape[1]):
        x[:,fearture_index]=standardize_vector(x[:,fearture_index])
    return x

'''
归一化一个特征向量
'''
def normalize_vector(x:np.ndarray)->np.ndarray:
    max=x.max()
    min=x.min()
    result=(x-min)/(max-min)
    return result

'''
归一化一个特征矩阵中的所有特征向量
'''
def normalize_matrix(x:np.ndarray)->np.ndarray:
    for fearture_index in range(x.shape[1]):
        x[:,fearture_index]=normalize_vector(x[:,fearture_index])
    return x


'''
将一个一维的特征向量映射成高维矩阵(不修改原特征矩阵)
'''
def reflect_to_high_dim(x:np.ndarray,column_index:int,high_dim_num:int)->np.ndarray:
    # x: 原特征矩阵 column_index: 要映射的特征的列索引 high_dim_num: 要扩展的次数 (大于等于2)
    reflect_feature=x[:,column_index]
    for i in range(2,high_dim_num+1):
        high_feature=np.power(reflect_feature,i)
        x = np.insert(x,column_index+1,high_feature,axis=1)
    return x
