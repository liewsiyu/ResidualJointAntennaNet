from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Best_subGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\101022\子信道增益(p=10行列).csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.2, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('信噪比为10时,ES_8822_子信道增益')
print(Best_subGain_Mean)