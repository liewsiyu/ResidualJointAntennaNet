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
import math
from time import time
from sklearn.model_selection import train_test_split
from numpy import mat
a = 10
I2 = np.eye(2)

x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]


dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\增益标签(p=10行列).csv')).iloc[:, 1:]
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
print("xTrain: ",len(xTrain))
print("xTest: ",len(xTest))


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
model.fit(xTrain,yTrain, n_epoch=100, shuffle=True, validation_set=(xTest, yTest),
          show_metric=False, batch_size=1280, snapshot_step=None,
          snapshot_epoch=False, run_id='vgg_oxflowers17')
train = time() - trainStart

accuracy = model.evaluate(xTest, yTest)  # 计算

n = 8
testStart = time()
VGG_Res = model.predict(xTest[0:40000])
test= time() - testStart
aaa = np.argmax(VGG_Res, axis=1) + 1  # 返回的只是索引，所以要+1
VGG_Res_np = np.array(aaa)  #
xTest_np = np.array(xTest[0:40000])

fullChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道增益(p=10行列).csv").iloc[:, 1:]
fullChainGain = fullChainGain[0:200000]
fullChainGain = np.asarray(fullChainGain, np.float32)

subChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道增益(p=10行列).csv").iloc[:, 1:]
subChainGain = subChainGain[0:200000]
subChainGain = np.asarray(subChainGain, np.float32)

fullChainGain_Mean = np.mean(fullChainGain)
subChainGain_Mean = np.mean(subChainGain)

"""测试阶段不止要测均值，也要测方差，时间
"""



Loss = []
Gain = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(VGG_Res_np[i])
    VGG_Sub = mat(np.zeros((2, 2)), dtype=float)
    VGG_Sub[0, 0] = ArrayA[i1, j1]
    VGG_Sub[0, 1] = ArrayA[i1, j2]
    VGG_Sub[1, 0] = ArrayA[i2, j1]
    VGG_Sub[1, 1] = ArrayA[i2, j2]

    Full_Gain = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    VGG_Gain = math.sqrt(1 / 2) * np.linalg.norm(VGG_Sub, ord='fro')
    Gain.append(VGG_Gain)
    Loss.append(Full_Gain - VGG_Gain)

Gain_Mean = np.mean(Gain)
Loss_Mean = np.mean(Loss)
Loss_Variance = np.var(Loss)

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




print("基于信道增益")
print("VGG(200000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道增益均值", fullChainGain_Mean)
print("测试样本的穷举法子信道增益均值", subChainGain_Mean)
print("穷举法损失均值", fullChainGain_Mean - subChainGain_Mean)
print('预测准确率', accuracy)
print('预测子信道增益均值', Gain_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)