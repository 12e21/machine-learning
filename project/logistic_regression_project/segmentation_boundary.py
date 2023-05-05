import numpy as np
import matplotlib.pyplot as plt

# 随机点阵生成器
axis1=np.random.randint(1,100,100)
axis2=np.random.randint(1,100,100)
results=np.zeros(100)
for i in range(100):
    if (axis1[i]-50)**2+(axis2[i]-50)**2 <800:
        results[i]=1


design_matrix=np.zeros([100,3])
design_matrix[:,0]=axis1
design_matrix[:,1]=axis2
design_matrix[:,2]=results
print(design_matrix)

plt.figure()
plt.scatter(design_matrix[:,0][np.where(results==1)],design_matrix[:,1][np.where(results==1)],c="red")
plt.scatter(design_matrix[:,0][np.where(results==0)],design_matrix[:,1][np.where(results==0)],c="blue")
plt.show()
