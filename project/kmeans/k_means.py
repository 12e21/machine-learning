import numpy as np
import matplotlib.pyplot as plt
def calculateDistance(point1:np.ndarray,point2:np.ndarray):
    power:np.ndarray
    power=np.power((point1-point2),2)
    return power.flatten().sum()


# choose two centroids
centroids=[np.array([[1.65,2.11]]),np.array([[0.61,1.53]]),np.array([[5.89,0.80]])]

data=np.loadtxt("data.txt")

assigns=[1]*data.shape[0]

fig=plt.figure()

for k in range(1000):
    # draw picture
    for m in range(data.shape[0]):
        if assigns[m]==0:
            plt.scatter(data[m,0],data[m,1],c="r")
        elif assigns[m]==1:
            plt.scatter(data[m, 0], data[m, 1], c="b")
        elif assigns[m]==2:
            plt.scatter(data[m, 0], data[m, 1], c="y")


    plt.show()
    plt.clf()

    # cluster assign
    for i in range(data.shape[0]):
        assign=0
        minVal=calculateDistance(data[i,:],centroids[0])
        for j in range(1,3):
            distance=calculateDistance(data[i,:],centroids[j])
            if distance<minVal:
                minVal=distance
                assign=j
        assigns[i]=assign

    centroidCount=[0,0,0]

    # move centroid
    for i in range(len(centroids)):
        centroids[i]=np.zeros(centroids[i].shape)

    for i in range(len(assigns)):
        centroids[assigns[i]]+=data[i,:]
        centroidCount[assigns[i]]+=1

    for i in range(len(centroids)):
        centroids[i]=centroids[i]/centroidCount[i]


