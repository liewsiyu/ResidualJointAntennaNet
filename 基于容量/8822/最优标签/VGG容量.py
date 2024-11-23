# -*- coding: utf-8 -*-
""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import pandas as pd
from time import time
from numpy import mat
from sklearn.model_selection import train_test_split

a = 50
x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]


dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=50行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)

label = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\容量标签(p=50行列).csv')).iloc[:, 1:]
label = np.asarray(label, np.int32)
label.astype(np.int32)

n_class = 784  # 组合数 784钟
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别

# 独热码赋值   如果label[i] = 3   label_array=[0,0,1,0.....0]
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1

# SVM用五折交叉验证的话，4:1
xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)



# Building 'VGG Network'
# network = input_data(shape=[None, 224, 224, 3])
network = input_data(shape=[None, 8, 8, 1])

network = conv_2d(network, 3, 1, activation='relu')
network = conv_2d(network, 3, 1, activation='relu')
# network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 5, 1, activation='relu')
network = conv_2d(network, 5, 1, activation='relu')
# network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 10, 1, activation='relu')
network = conv_2d(network, 10, 1, activation='relu')
network = conv_2d(network, 10, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 20, 1, activation='relu')
network = conv_2d(network, 20, 1, activation='relu')
network = conv_2d(network, 20, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 32, 1, activation='relu')
network = conv_2d(network, 32, 1, activation='relu')
network = conv_2d(network, 32, 1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, n_class, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
trainStart = time()
model.fit(xTrain, yTrain, n_epoch=100, shuffle=True, validation_set=(xTest, yTest),
          show_metric=False, batch_size=1280, snapshot_step=None,
          snapshot_epoch=False, run_id='vgg_oxflowers17')
train = time() - trainStart


# 计算
accuracy = model.evaluate(xTest, yTest)


testStart = time()
VGG_Res = model.predict(xTest[0:40000])
test= time() - testStart

aaa = np.argmax(VGG_Res, axis=1) + 1  # 返回的只是索引，所以要+1
VGG_Res_np = np.array(aaa)  #
xTest_np = np.array(xTest[0:40000])


fullChannelCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道容量(p=50行列).csv").iloc[:, 1:]
fullChannelCapacity = fullChannelCapacity [0:200000]
fullChannelCapacity = np.asarray(fullChannelCapacity, np.float32)

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道容量(p=50行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)


fullChannelCapacity_Mean = np.mean(fullChannelCapacity )
Best_subCapacity_Mean = np.mean(Best_subCapacity)



I1 = np.eye(8)
I2 = np.eye(2)
Pre_Loss = []
Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(VGG_Res_np[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    subCapacity = np.log2(np.linalg.det(I2 + a * Pre_sub.T * Pre_sub / 2))

    Pre_Capacity.append(subCapacity)
    Pre_Loss.append(fullCapacity - subCapacity)

Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)


if test < 1e-6:
    testUnit = "ns"
    test *= 1e9
elif test < 1e-3:
    testUnit = "us"
    test *= 1e6
elif test < 1:
    testUnit = "ms"
    test *= 1e3
else:
    testUnit = "s"

if train < 1e-6:
    trainUnit = "ns"
    train *= 1e9
elif train < 1e-3:
    trainUnit = "us"
    train *= 1e6
elif train < 1:
    trainUnit = "ms"
    train *= 1e3
else:
    trainUnit = "s"


print("基于信道容量 信噪比: ", a)
print("VGG(40000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道容量均值", fullChannelCapacity_Mean)
print("测试样本的穷举法子信道容量均值",Best_subCapacity_Mean)
print("穷举法损失均值", fullChannelCapacity_Mean - Best_subCapacity_Mean)
print('预测准确率', accuracy)
print('预测子信道容量均值', Capacity_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)

