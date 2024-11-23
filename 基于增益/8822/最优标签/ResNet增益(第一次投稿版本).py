from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
import pandas as pd
from numpy import mat
import math
from time import time
from sklearn.model_selection import train_test_split
import tensorflow as tf


# 快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)

matrix = x.values

def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]


a = 10
# 对的
dataset = pd.read_csv(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv').iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(r'D:\天线选择 郭志斌\正确数据集\增益标签(p=10行列).csv').iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)

n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1

#随机划分
xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ",len(xTrain))
print("xTest: ",len(xTest))


# Building Residual Network
net = tflearn.input_data(shape=[None, 8, 8, 1])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, n_class, activation='softmax',regularizer='L2')
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                    max_checkpoints=10, tensorboard_verbose=0)
trainStart = time()
model.fit(xTrain, yTrain, n_epoch=100, shuffle=True, validation_set=(xTest, yTest),
          show_metric=True,snapshot_step=200, snapshot_epoch=False,batch_size=1280, run_id='resnet_mnist')
train = time() - trainStart


# 计算
accuracy = model.evaluate(xTest, yTest)

n=8

#你要计算预测的时间，应该是要放在predict部分的
testStart = time()
Residuel_Pre = model.predict(xTest[0:80000])
test = time() - testStart


aaa=np.argmax(Residuel_Pre,axis=1)+1
Residuel_Pre_np = np.array(aaa)
xTest_np = np.array(xTest[0:80000])


fullChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\40w全信道增益(p=10行列).csv").iloc[:, 1:]
fullChainGain = fullChainGain[0:400000]
fullChainGain = np.asarray(fullChainGain, np.float32)

subChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\40w子信道增益(p=10行列).csv").iloc[:, 1:]
subChainGain = subChainGain[0:400000]
subChainGain = np.asarray(subChainGain, np.float32)

fullChainGain_Mean = np.mean(fullChainGain)
subChainGain_Mean = np.mean(subChainGain)

"""测试阶段不止要测均值，也要测方差，时间
"""



I = np.eye(n)
I2 = np.eye(2)
Loss = []
Gain = []
for i in range(80000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(Residuel_Pre_np[i])   #通过其反推 ij
    ResNet_Sub = mat(np.zeros((2, 2)), dtype=float)
    ResNet_Sub[0, 0] = ArrayA[i1, j1]
    ResNet_Sub[0, 1] = ArrayA[i1, j2]
    ResNet_Sub[1, 0] = ArrayA[i2, j1]
    ResNet_Sub[1, 1] = ArrayA[i2, j2]
    Pre_fullGian = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    Pre_subGian = math.sqrt(1 / 2) * np.linalg.norm(ResNet_Sub, ord='fro')
    Gain.append(Pre_subGian)
    Loss.append(Pre_fullGian-Pre_subGian)


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
print("ResNet(400000个样本)")
print("320000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("80000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道增益均值", fullChainGain_Mean)
print("测试样本的穷举法子信道增益均值", subChainGain_Mean)
print("穷举法损失均值", fullChainGain_Mean - subChainGain_Mean)
print('预测准确率', accuracy)
print('预测子信道增益均值', Gain_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)