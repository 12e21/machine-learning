import numpy as np

'''
本文件存放一些工具函数
'''

'''
标准化一个正则向量
'''
def standardize_vector(x:np.ndarray)->np.ndarray:
    mean=np.mean(x)
    std=np.std(x)
    result=(x-mean)/std
    return result

'''
归一化一个特征向量
'''
def normalize_vector(x:np.ndarray)->np.ndarray:
    max=x.max()
    min=x.min()
    result=(x-min)/(max-min)
    return result


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




if __name__ == "__main__":
    #test_arr=np.array([1234,5673,8765])
    #print(standardize_vector(test_arr))
    #print(normalize_vector(test_arr))
    test_matrix=np.array([
        [1,100,1000],
        [2,200,2000],
        [3,300,3000]
    ])
    print(reflect_to_high_dim(test_matrix,0,2))
    print(test_matrix)