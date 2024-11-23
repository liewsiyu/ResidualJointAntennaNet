# 这是运行结果截图的程序,最初的一版，一直没改过
# 因为文章中使用的关键数据   accuracy + 预测的mean capacity + 运行时间部分的代码没啥毛病, 就没重新再改重新跑了
# 所以截图只用了上述三个数据, 其它用另外的程序重新跑了，详见下面
# 如果想整理代码, 整成ResNet容量(返修版本).py重新跑


from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import pandas as pd
from time import time
from numpy import mat
from sklearn.model_selection import train_test_split



# 快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]


# 信噪比
a = 50
n = 8
X_orig = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=50行列).csv").iloc[:, 1:]
X_orig = np.asarray(X_orig, np.float32)
dataset = X_orig.reshape(X_orig.shape[0],8,8,1)

label = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\容量标签(p=50行列).csv").iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)
n_class =  784


# 标签转换成独热编码
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))   # 样本，类别
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1

# 划分数据集4:1  随机数random_state=40
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

trainStart = time()
model.fit(xTrain, yTrain, n_epoch=100, validation_set=(xTest, yTest), shuffle=True,
          show_metric=True, batch_size=1280, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
train = time() - trainStart


# 计算
accuracy = model.evaluate(xTest, yTest)

testStart = time()
AlexNet_Predict = model.predict(xTest[0:40000])
test= time() - testStart

aaa = np.argmax(AlexNet_Predict , axis=1) + 1  # 返回的只是索引，所以要+1
AlexNet_Predict_np = np.array(aaa)
xTest_np = np.array(xTest[0:40000])



# 下面的代码是老版本的代码, 也就是结果截图中运行程序的代码
# 如果你想重新跑的话，建议更新成 /基于容量/8822/最优标签/ResNet(返修版本).py文件中相同位置的代码
#         |                                                  (这个是迭代过几次的, 比较清晰)
#         |
#    |    |    |
#       | |  |
#         |

# 最初采用全部随机,全信道和穷举子信道全用来求平均值,但是实际不合理, 因此我们使用的是4w个样本而已
# 所以在DSP的修回稿中, 我已经重新更新了  Best_subCapacity_Mean最佳子信道容量
# 详见本文件目录下 获取最优ES_Capacity.py
# -------------------------------------------------------------------------------------------
fullCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道容量(p=50行列).csv").iloc[:, 1:]
fullCapacity = fullCapacity[0:200000]
fullCapacity  = np.asarray(fullCapacity, np.float32)
Best_subCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道容量(p=50行列).csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
# 全信道容量文章里面没用到
fullCapacity_Mean = np.mean(fullCapacity)
# ES算法的最佳子信道容量 获取最优ES_Capacity.py来得到
Best_subCapacity_Mean = np.mean(Best_subCapacity)
# -------------------------------------------------------------------------------------------


# fullChainCapacity_Mean  subChainCapacity_Mean有上面那块提到的问题, 下面使用这两个变量计算出来的结果都有问题
# 但是有问题的结果没用上, 你可以选择忽略 or  重新整理成ResNet容量(返修版本那样重新跑)
# 或者你可以看看xlsx文件, 我把有问题的行删了，但是剩下来的部分可以对应截图文件看
# 简单来说,就是截图的数据比xlsx文件的数据多了一点

I1 = np.eye(8)
I2 = np.eye(2)
Pre_Loss = []
Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(AlexNet_Predict_np[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    Pre_fullCapacity =  np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity = np.log2(np.linalg.det(I2 + a * Pre_sub.T * Pre_sub / 2))
    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)

# 建议更新成 /基于容量/8822/最优标签/ResNet(返修版本).py文件中相同位置的代码
# 代码冗余
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


print("基于信道容量 信噪比 ", a)
print("AlexNet(200000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
#没用到
print("测试样本的全信道容量均值", fullCapacity_Mean)
#用另外的程序重新更新了   获取最优ES_Capacity.py
print("测试样本的穷举法子信道容量均值",Best_subCapacity_Mean)
#没用到
print("穷举法损失均值", fullCapacity_Mean - Best_subCapacity_Mean)

print('预测准确率', accuracy)
print('预测子信道容量均值', Capacity_Mean)

#没用到
print('预测损失均值', Loss_Mean)
#没用到
print('预测损失方差', Loss_Variance)

