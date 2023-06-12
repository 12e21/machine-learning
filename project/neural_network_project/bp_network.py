import numpy as np
'''
Summary:
    this class is a back propagation neural network.
    you can set the layer count and neural count of every layer of input, output and hidden layer.
'''
class BpNetwork:
    def __init__(self,feature:np.ndarray=None,label:np.ndarray=None) -> None:
        self.feature=feature
        self.label=label