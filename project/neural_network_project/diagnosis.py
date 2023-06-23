from bp_network import * 
from machine_learn import utils
import matplotlib.pyplot as plt

class NetworkDiagnosis():
    def __init__(self,data_feature:np.ndarray,data_label:np.ndarray,network:BpNetwork) -> None:
            self.data_feature=data_feature
            self.data_label=data_label
            self.network=network
            self.trainSetFeature=None
            self.trainSetLabel=None
            self.validateSetFeature=None
            self.validateSetLabel=None
            self.testSetFeature=None
            self.testSetLabel=None

    def splitDataSet(self):
            dataSet=np.concatenate((self.data_feature,self.data_label),axis=1)
            np.random.shuffle(dataSet)

            setSequence=np.split(dataSet,[int(dataSet.shape[0]*0.6),int(dataSet.shape[0]*0.8)],axis=0)
            trainSet=setSequence[0]
            validateSet=setSequence[1]
            testSet=setSequence[2]

            self.trainSetFeature=trainSet[:,:-self.data_label.shape[1]]
            self.trainSetLabel=trainSet[:,-self.data_label.shape[1]:].reshape([trainSet.shape[0],self.data_label.shape[1]])
            
            self.validateSetFeature=validateSet[:,:-self.data_label.shape[1]]
            self.validateSetLabel=validateSet[:,-self.data_label.shape[1]:].reshape([validateSet.shape[0],self.data_label.shape[1]])

            self.testSetFeature=testSet[:,:-self.data_label.shape[1]]
            self.testSetLabel=testSet[:,-self.data_label.shape[1]:].reshape([testSet.shape[0],self.data_label.shape[1]])

    def train(self,iterateCount=10000)->float:
        self.network.feature=self.trainSetFeature.transpose()
        self.network.label=self.trainSetLabel.transpose()
        self.network.train(iterateCount,ifShowLoss=False)
        #print("train bias: "+str(self.network.loss[1]))
        return self.network.loss[1]

    def validate(self)->float:
        self.network.feature=self.validateSetFeature.transpose()
        self.network.label=self.validateSetLabel.transpose()
        self.network.one_iterate(if_gradient_drease=False)
        #print("validate variance: "+str(self.network.loss[1]))
        return self.network.loss[1]

    def test(self)->float:
        self.network.feature=self.testSetFeature.transpose()
        self.network.label=self.testSetLabel.transpose()
        self.network.one_iterate(if_gradient_drease=False)
        np.savetxt("project/neural_network_project/test/label.txt",self.network.label.transpose())
        np.savetxt("project/neural_network_project/test/predict.txt",self.network.predict_result.transpose())
        #print("test loss: "+str(self.network.loss[1]))
        return self.network.loss[1]


    def drawCurve(self):
        '''
        @draw the learning diagnosis curve
        '''
        self.splitDataSet()

        trainFeatureCopy=self.trainSetFeature.copy()
        trainLabelCopy=self.trainSetLabel.copy()

        trainLossList=[]
        crossOverLossList=[]


        
        for i in range(10,trainFeatureCopy.shape[0],5):
             self.trainSetFeature=trainFeatureCopy[:i,:]
             self.trainSetLabel=trainLabelCopy[:i,:]

             trainLossList.append(self.train(iterateCount=1000))
             crossOverLossList.append(self.validate())
             print(str(i)+"/"+str(trainFeatureCopy.shape[0]))

        self.data_feature=trainFeatureCopy
        self.data_label=trainLabelCopy
        plt.figure()
        plt.plot(range(10,trainFeatureCopy.shape[0],5),trainLossList,range(10,trainFeatureCopy.shape[0],5),crossOverLossList)
        plt.show()

             

        

            
         
    
    

if __name__ == "__main__":

    # load data
    feature=np.loadtxt("project/iris_sort/data/feature.txt")
    label=np.loadtxt("project/iris_sort/data/label.txt")
    
    # normalize the data, this part is especilly important , if not,the loss could be decreasing very slowly
    feature=utils.normalize_matrix(feature)
    net1=BpNetwork(feature.transpose(),label.transpose(),3,[5,10,3])
    net1.learning_rate=0.3
    net1.regular_rate=0.0

    '''
    # this part is for look the result of predict
    net1.train(2000)
    np.savetxt("project/neural_network_project/test/label.txt",net1.label.transpose())
    np.savetxt("project/neural_network_project/test/predict.txt",net1.predict_result.transpose())
    '''

    '''
    Diag1=NetworkDiagnosis(data_feature=feature,data_label=label,network=net1)
    Diag1.splitDataSet()
    Diag1.train()
    Diag1.validate()
    Diag1.test()
    '''

    Diag1=NetworkDiagnosis(data_feature=feature,data_label=label,network=net1)
    Diag1.drawCurve()


