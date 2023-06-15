from bp_network import * 
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

    def train(self):
        self.network.feature=self.trainSetFeature.transpose()
        self.network.label=self.trainSetLabel.transpose()
        self.network.train(5000,False)
        print("train bias: "+str(self.network.loss[1]))

    def validate(self):
        self.network.feature=self.validateSetFeature.transpose()
        self.network.label=self.validateSetLabel.transpose()
        self.network.one_iterate(if_gradient_drease=False)
        print("validate variance: "+str(self.network.loss[1]))

    def test(self):
        self.network.feature=self.testSetFeature.transpose()
        self.network.label=self.testSetLabel.transpose()
        self.network.one_iterate(if_gradient_drease=False)
        print("test loss: "+str(self.network.loss[1]))
         
    
    

if __name__ == "__main__":
    net1=BpNetwork(np.array([
    [0,0,0,1,1,1,0,1],
    [0,0,1,0,1,0,1,1],
    [0,1,0,0,0,1,1,1],
]),np.array([
    [0,0,0,0,1,1,0,1],
    [1,1,1,1,0,0,1,0]
    ]),4,[3,4,3,2])

    Diag1=NetworkDiagnosis(data_feature=np.array([
    [0,0,0,1,1,1,0,1],
    [0,0,1,0,1,0,1,1],
    [0,1,0,0,0,1,1,1],
]).transpose(),data_label=np.array([
    [0,0,0,0,1,1,0,1],
    [1,1,1,1,0,0,1,0]
    ]).transpose(),network=net1)

    Diag1.splitDataSet()
    Diag1.train()
    Diag1.validate()
    Diag1.test()
