import numpy as np
'''
Summary:
    this class is a back propagation neural network.
    you can set the layer count and neural count of every layer of input, output and hidden layer.
'''
class BpNetwork:
    def __init__(self,feature:np.ndarray,label:np.ndarray,layer_count:int,neural_count_of_layers:tuple) -> None:
        
        # get some params
        self.feature=feature
        self.label=label
        self.layer_count=layer_count
        self.neural_count_of_layers=neural_count_of_layers

        # init network
        self.activate_item=[np.zeros([i,1]) for i in neural_count_of_layers]
        self.z_item=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]
        self.weights=[np.random.rand(neural_count_of_layers[i+1],neural_count_of_layers[i]) for i in range(layer_count-1)]
        self.bias_weights=[np.random.rand(i,1) for i in neural_count_of_layers[1:]]
        self.thetas=[np.concatenate((self.weights[i],self.bias_weights[i]),1) for i in range(layer_count-1)]
        self.epsilons=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]
        self.deltas=[np.zeros([neural_count_of_layers[i+1],neural_count_of_layers[i]]) for i in range(layer_count-1)]
        self.bias_deltas=[np.zeros([i,1]) for i in neural_count_of_layers[1:]]

        self.loss=[0,0]
        self.learning_rate = 0.3
        self.regular_rate = 0.00000001


    def sigmoid(self,x:np.ndarray):
        '''
        activate function
        '''
        return 1/(1+np.exp(-x))
    


    def forward_propagation(self):
        '''
        forward propagation for one data
        '''
        for i in range(self.layer_count-1):
            a_bias=np.concatenate((self.activate_item[i],[[1]]),0)
            self.z_item[i]=np.dot(self.thetas[i],a_bias)
            self.activate_item[i+1]=self.sigmoid(self.z_item[i])


    def cal_loss(self,current_label:np.ndarray):
        '''
        calculate loss for one data
        '''
        self.loss[0]+=(current_label*np.log(self.activate_item[-1])+(1-current_label)*np.log(1-self.activate_item[-1])).flatten().sum()


    def back_propagation(self,current_label:np.ndarray):
        '''
        back propagation for one data
        '''
        self.epsilons[-1]=self.activate_item[-1]-current_label
        for i in range(self.layer_count-3,-1,-1):
            self.epsilons[i]=np.dot(self.weights[i+1].transpose(),self.epsilons[i+1])*(self.activate_item[i+1]*(1-self.activate_item[i+1]))

            
    def cal_delta(self):
        '''
        calculate delta for one data
        '''
        for i in range(self.layer_count-2,-1,-1):
            self.deltas[i]+=np.dot(self.epsilons[i],self.activate_item[i].transpose())
            self.bias_deltas[i]+=self.epsilons[i]


    def cal_all(self,current_feature:np.ndarray,current_label:np.ndarray):
        '''
        whole process of bp network calculate for one data
        '''
        self.activate_item[0]=current_feature
        self.forward_propagation()
        self.cal_loss(current_label)
        self.back_propagation(current_label)
        self.cal_delta()


    def gradient_decrease(self):
        '''
        accumulate gradient for every data and decrease the gradient
        '''
        for i in range(self.layer_count-1):
            self.weights[i]-=((1./self.feature.shape[1])*self.learning_rate*self.deltas[i]+self.regular_rate*self.weights[i])
            self.bias_weights[i]-=(1./self.feature.shape[1])*self.learning_rate*self.bias_deltas[i]
        

        self.thetas=[np.concatenate((self.weights[i],self.bias_weights[i]),1) for i in range(self.layer_count-1)]



    def one_iterate(self,if_gradient_drease:bool=True):
        '''
        one whole iterate,return loss
        '''
        for i in range(self.label.shape[1]):
            self.cal_all(self.feature[:,i].reshape([self.feature[:,i].size,1]),self.label[:,i].reshape([self.label[:,i].size,1]))
        if if_gradient_drease:
            self.gradient_decrease()

        self.loss[1]=self.loss[0]*(-1./self.feature.shape[1])+(self.regular_rate/(2.*self.feature.shape[1]))*sum([(w**2).flatten().sum() for w in self.weights])
        self.loss[0]=0
        self.deltas=[np.zeros([self.neural_count_of_layers[i+1],self.neural_count_of_layers[i]]) for i in range(self.layer_count-1)]
        self.bias_deltas=[np.zeros([i,1]) for i in self.neural_count_of_layers[1:]]


    def train(self,iterateCount:int,ifShowLoss:bool=True):
        for i in range(iterateCount):
            self.one_iterate()
            if ifShowLoss==True :
                print(self.loss[1])


if __name__ == "__main__":

    net1=BpNetwork(np.array([
    [0,0,0,1,1,1,0,1],
    [0,0,1,0,1,0,1,1],
    [0,1,0,0,0,1,1,1],
]),np.array([
    [0,0,0,0,1,1,0,1],
    [1,1,1,1,0,0,1,0]
    ]),4,[3,4,3,2])

    net1.train(10000)