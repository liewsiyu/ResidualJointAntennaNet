from __future__ import division, print_function, absolute_import
from numpy import mat
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from numpy import mat
import pandas as pd
from time import time
import math
from sklearn.model_selection import train_test_split

# 快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]



X_orig = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv").iloc[:, 1:]
X_orig = np.asarray(X_orig, np.float32)
dataset = X_orig.reshape(X_orig.shape[0],8,8,1)

label = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\增益标签(p=10行列).csv").iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别
for i in range(n_sample):       #将数据转化为独热编码 1 [1,0,0...0,0,0,0]
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1



xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ",len(xTrain))
print("xTest: ",len(xTest))

# Building 'AlexNet'
network = input_data(shape=[None, 8, 8, 1])
network = conv_2d(network, 4, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 12, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 48, 3, activation='relu')
network = conv_2d(network, 48, 3, activation='relu')
network = conv_2d(network, 36, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 576, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 576, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, n_class, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
trainStart = time()   #计算训练时长
model.fit(xTrain, yTrain, n_epoch=100, validation_set=(xTest, yTest), shuffle=True,
          show_metric=True, batch_size=1280, snapshot_step=None,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
train = time() -trainStart

# 计算
accuracy = model.evaluate(xTest, yTest)

n = 8
testStart = time()
AlexNet_Pre = model.predict(xTest[0:40000])
test = time() - testStart              #测试20000个样本的时间

aaa = np.argmax(AlexNet_Pre, axis=1) + 1  # 返回的只是索引, 所以要+1
AlexNet_Pre_np = np.array(aaa)  #
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

    i1, i2, j1, j2 = GetIndexFrom(AlexNet_Pre_np[i])
    #不要用np.mat([[0, 0], [0, 0]]), float会直接被转成int型
    Alex_Sub = mat(np.zeros((2, 2)), dtype=float)
    Alex_Sub[0, 0] = ArrayA[i1, j1]
    Alex_Sub[0, 1] = ArrayA[i1, j2]
    Alex_Sub[1, 0] = ArrayA[i2, j1]
    Alex_Sub[1, 1] = ArrayA[i2, j2]

    Full_Gain = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    Alex_Gain = math.sqrt(1 / 2) * np.linalg.norm(Alex_Sub, ord='fro')
    Gain.append(Alex_Gain)
    Loss.append(Full_Gain - Alex_Gain)

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
print("AleNet(200000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道增益均值", fullChainGain_Mean)
print("测试样本的穷举法子信道增益均值", subChainGain_Mean)
print("穷举法损失均值", fullChainGain_Mean - subChainGain_Mean)
print('预测准确率', accuracy)
print('预测子信道增益均值', Gain_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)


