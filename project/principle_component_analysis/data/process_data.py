import pandas as pd
import numpy as np
from utils import standardize_matrix

data=pd.read_csv(filepath_or_buffer="winequality-red.csv")
# 标准化数据
data=standardize_matrix(data.values)
np.savetxt("feature.txt",data[:,:-1])