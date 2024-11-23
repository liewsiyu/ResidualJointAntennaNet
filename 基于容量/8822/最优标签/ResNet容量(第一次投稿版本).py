from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
from numpy import mat
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))


#快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]


a = 10
#数据预处理
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\容量标签(p=10行列).csv')).iloc[:, 1]
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
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False )
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.activation(net, 'relu')
net = tflearn.batch_normalization(net)
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, n_class, activation='softmax',regularizer='L2')
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                    max_checkpoints=20, tensorboard_verbose=0)
trainStart = time()
model.fit(xTrain, yTrain, n_epoch=100, shuffle=True, validation_set=(xTest, yTest),
          show_metric=True,snapshot_step=200, snapshot_epoch=False,batch_size=1280, run_id='resnet_mnist')
train = time() - trainStart

# 计算
accuracy = model.evaluate(xTest, yTest)


#你要计算预测的时间，应该是要放在predict部分的
testStart = time()
ResNet_Pre = model.predict(xTest[0:40000])
test = time() - testStart


aaa=np.argmax(ResNet_Pre,axis=1)+1
ResNet_Pre_np = np.array(aaa)


xTest_np = np.array(xTest[0:40000])


fullCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道容量(p=10行列).csv").iloc[:, 1:]
fullCapacity = fullCapacity[0:200000]
fullCapacity = np.asarray(fullCapacity , np.float32)

Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道容量(p=10行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)

fullCapacity_Mean = np.mean(fullCapacity)
Best_subCapacity_Mean = np.mean(Best_subCapacity)



I1 = np.eye(8)
I2 = np.eye(2)
Pre_Loss = []
Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np[i])  # 通过其反推 ij
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 2))

    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


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

print("基于信道容量 信噪比 ",a)
print("ResNet(200000个样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道容量均值", fullCapacity_Mean)
print("测试样本的穷举法子信道容量均值",Best_subCapacity_Mean)
print("穷举法损失均值", fullCapacity_Mean - Best_subCapacity_Mean)
print('预测准确率', accuracy)
print('预测子信道容量均值', Capacity_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)

