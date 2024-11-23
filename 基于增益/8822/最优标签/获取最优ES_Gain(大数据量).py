from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Best_subGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\8822\子信道增益(p=10行列).csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.2, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print("保留4位小数")
print('ES 20w 8822 gain')
print(Best_subGain_Mean)

Best_subGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\ResNet 大数据量对比\40w子信道增益(p=10行列).csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:400000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.2, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('ES 40w 8822 gain')
print(Best_subGain_Mean)

Best_subGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\ResNet 大数据量对比\60w子信道增益(p=10行列).csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:600000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.2, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('ES 60w 8822 gain')
print(Best_subGain_Mean)


