from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from uitls_101022_capcacity import computation_time

#快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\天线选择 郭志斌\正确数据集\101022\101022快速查询数据表.csv', header=None)
matrix = x.values
# 56 * 56 =  2025
def GetIndexFrom_101022(y_pre):
    for i in range(0,2025):
        if y_pre == matrix[i][0]:
            # i1 i2  j1 j2
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]


a = 40
# 数据预处理
# iloc[:,1:].values   #从第2列到最后(因为一开始会有一个索引)
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\101022\全信道矩阵(p=40行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 10, 10, 1)
# .iloc[:, 1]  只取第2列(因为一开始会有一个索引))
label = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\101022\JTRAS容量标签(p=40行列).csv')).iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)
# 10选2  10选2  所以就是  45 * 45 = 2025
# 标签转换成独热编码
n_class = 2025
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1

# 随机划分
xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ",len(xTrain))
print("xTest: ",len(xTest))




# Building Residual Network
net = tflearn.input_data(shape=[None, 10, 10, 1])
net = tflearn.conv_2d(net, 64, 2, activation='relu', bias=True, regularizer='L2', weight_decay=0.0001)
net = tflearn.conv_2d(net, 128, 2, activation='relu', bias=True, regularizer='L2', weight_decay=0.0001)
net = tflearn.conv_2d(net, 256, 2, activation='relu', bias=True , regularizer='L2', weight_decay=0.0001)
net = tflearn.conv_2d(net, 128, 2, activation='relu', bias=True, regularizer='L2', weight_decay=0.0001)
net = tflearn.conv_2d(net, 64, 2, activation='relu', bias=True, regularizer='L2', weight_decay=0.0001)
# Residual blocks
net = tflearn.residual_bottleneck(net, 2, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.activation(net, 'relu')
net = tflearn.batch_normalization(net)
net = tflearn.global_avg_pool(net)
# Regression
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)

# Regression
net = tflearn.fully_connected(net, n_class, activation='softmax')
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                    max_checkpoints=20, tensorboard_verbose=0)
trainStart = time()
model.fit(xTrain, yTrain, n_epoch=60, shuffle=True, validation_set=(xTest, yTest),
          show_metric=True,snapshot_step=None, snapshot_epoch=False,batch_size=512, run_id='resnet_mnist')
train = time() - trainStart

# 计算
accuracy = model.evaluate(xTest, yTest)


# 你要计算预测的时间，应该是要放在predict部分的
# 拆成两部分预测是因为GPU的显存不够
testStart = time()
ResNet_Pre1 = model.predict(xTest[0:20000])
ResNet_Pre2 = model.predict(xTest[20000:40000])
test = time() - testStart

aaa1=np.argmax(ResNet_Pre1,axis=1)+1
aaa2=np.argmax(ResNet_Pre2,axis=1)+1
ResNet_Pre_np1 = np.array(aaa1)
ResNet_Pre_np2 = np.array(aaa2)
# 将两部分预测的结果进行拼接
# [1,2,3,4]  [5,6,7,8]
# [1,2,3,4,5,6,7,8]
ResNet_Pre_np = np.hstack((ResNet_Pre_np1, ResNet_Pre_np2))


xTest_np = np.array(xTest[0:40000])


I1 = np.eye(10)
I2 = np.eye(2)
#预测的信道容量损失
Pre_Loss = []
#预测的子信道容量
Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(10, 10)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1,j2 = GetIndexFrom_101022(ResNet_Pre_np[i])  # 通过其反推 ij
    Pre_sub = ArrayA[[i1, i2]][:, [j1, j2]]

    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 10))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 2))

    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)



print("基于信道容量 信噪比 ",a)
print("101022_Capacity_ResNet_SubOptimalLabel(200000个样本)")
print("160000个样本的训练时间 %.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000个样本的测试时间 %.1f %s" % (computation_time(test)[0], computation_time(test)[1]))
print('预测准确率', accuracy)
print('预测子信道容量均值', Capacity_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)

