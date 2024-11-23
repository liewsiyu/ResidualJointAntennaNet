from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\101022\子信道容量(p=10行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为10时,ES_101022_子信道容量')
print(Best_subCapacity_Mean)



Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\101022\子信道容量(p=20行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为20时,ES_101022_子信道容量')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\101022\子信道容量(p=30行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为30时,ES_101022_子信道容量')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\101022\子信道容量(p=40行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为40时,ES_101022_子信道容量')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\101022\子信道容量(p=50行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('信噪比为50时,ES_101022_子信道容量')
print(Best_subCapacity_Mean)



