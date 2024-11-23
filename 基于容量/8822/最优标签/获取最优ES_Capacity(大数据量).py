from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\ResNet 大数据量对比\40w子信道容量(p=10行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:400000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为10时,40w_ES_8822_子信道容量')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\ResNet 大数据量对比\60w子信道容量(p=10行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:600000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为20时,60w_ES_8822_子信道容量')
print(Best_subCapacity_Mean)






