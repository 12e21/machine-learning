import numpy as np
from data.utils import ensure_path_sep


def calVarianceLoss(feature:np.ndarray,featureApprox:np.ndarray)->float:
    """
    计算数据变化损失率
    """
    return (np.power((feature-featureApprox),2).flatten().sum())/(np.power((feature),2).flatten().sum())

if __name__ == "__main__":
    # 获取数据
    feature =np.loadtxt(ensure_path_sep("data/feature.txt"))

    dataCount=feature.shape[0]
    featureCount=feature.shape[1]

    # 计算协方差矩阵
    covarianceMatrix=np.zeros([featureCount,featureCount])
    for i in range(dataCount):
        covarianceMatrix+=np.outer(feature[i,:],feature[i,:])
    covarianceMatrix/=dataCount

    # 对协方差矩阵进行奇异值分解
    u:np.ndarray
    s:np.ndarray
    v:np.ndarray
    u,s,v=np.linalg.svd(covarianceMatrix)

    # 将特征降低到不同维度，计算相应量
    reduceMatrix = np.ndarray([0, 0])
    z=np.ndarray([0,0])
    xApprox=np.ndarray([0,0])

    for k in range(featureCount):
        # 计算降维矩阵
        reduceMatrix:np.ndarray=u[:,:k+1]
        # 计算降维后的结果
        z=np.dot(feature,reduceMatrix)
        # 计算降维后再升维的近似结果
        xApprox=np.dot(z,reduceMatrix.transpose())
        # 计算数据损失率
        varianceLoss=calVarianceLoss(feature=feature,featureApprox=xApprox)
        print("k为{0}时,变化保留率为{1}".format(k,varianceLoss))

        if varianceLoss<0.01:
            break

        # 储存降维矩阵
        np.savetxt("reduce_matrix.txt",reduceMatrix)







