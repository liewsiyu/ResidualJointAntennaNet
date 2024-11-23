import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  time import time
import math
from numpy import mat
x = pd.read_csv(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]

#100000
TotalbachSize =  100000
TestPercent = 0.2
TrainLoop =  50000
TotalTrainSize = TotalbachSize * (1 - TestPercent)

print('start')
a = 10
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv')).iloc[0:200000, 1:]
label = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\增益标签(p=10行列).csv')).iloc[0:200000, 1:]


dataset = np.asarray(dataset, np.float32)
label = np.asarray(label, np.int32)
label.astype(np.int32)

n_class = 784
print(n_class)
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别
for i in range(n_sample):
    label_array[i, label[i]-1] = 1  # 非零列赋值为1





xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=TestPercent, random_state=40)




x = tf.placeholder(tf.float32, [None, 64])  #输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, n_class])


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 构建网络
x_image = tf.reshape(x, [-1, 8, 8, 1])  # 转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
print(("h_conv1's shape is", (h_conv1.shape)))
h_pool1 = max_pool(h_conv1)  # 第一个池化层
print(("h_pool1's shape is", (h_pool1.shape)))

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
print(("h_conv2's shape is", (h_conv2.shape)))
h_pool2 = max_pool(h_conv2)  # 第二个池化层
print(("h_pool2's shape is", (h_pool2.shape)))

W_fc1 = weight_variable([2 * 2 * 64, 1024])
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 64])  # reshape成向量
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层
print(("h_fc1's shape is", (h_fc1.shape)))
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=0.4)  # dropout层
print(("h_fc1_drop's shape is", (h_fc1_drop.shape)))

W_fc2 = weight_variable([1024, n_class])  # 10
b_fc2 = bias_variable([n_class])  # 10
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层
cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# y_predictResult = y_predict.eval(feed_dict={x: xTest, keep_prob: 1.0})

batch_size = 128


trainStart = time()
for i in range(100000):
    #start = (i * batch_size) % (95000)
    #end = ((i * batch_size) + batch_size - 1) % (95000)


    # start = (i * batch_size) % dataset_size
    # end = min(start + batch_size, dataset_size)
    start = (i * batch_size) % 200000
    end = min(start + batch_size, 200000)
    if i % 1000 == 0:
        print(i)
        train_acc = sess.run(accuracy, feed_dict={x: xTrain[start:end], y_actual: yTrain[start:end], keep_prob: 1.0})
        print(train_acc)
    #         # print(sess.run(correct_prediction))
    #         print('step %d, training accuracy %g' % (i, train_acc))
    #
    #     # train_step.run(feed_dit={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
    train_step.run(feed_dict={x: xTrain[start:end], y_actual: yTrain[start:end], keep_prob: 0.5})
# y_predict.eval(feed_dict={x: xTest, y_actual: yTest, keep_prob: 0.5})
train = time() - trainStart


test_acc = accuracy.eval(feed_dict={x: xTest, y_actual: yTest, keep_prob: 1.0})
# y_predictResult =y_predict.eval(feed_dict={x: xTest,keep_prob: 1.0})
print("test accuracy %g" % test_acc)



testStart = time()
y_PreRes = y_predict.eval(feed_dict={x: xTest[0:40000], keep_prob: 1.0})
test = time() - testStart

y_PreRes_np = np.array(y_PreRes)
print(len(y_PreRes_np))
y_PreRes_np_Res = np.argmax(y_PreRes_np, axis=1)+1



fullChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道增益(p=10行列).csv").iloc[:, 1:]
fullChainGain = fullChainGain[0:200000]
fullChainGain = np.asarray(fullChainGain, np.float32)

subChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道增益(p=10行列).csv").iloc[:, 1:]
subChainGain = subChainGain[0:200000]
subChainGain = np.asarray(subChainGain, np.float32)

fullChainGain_Mean = np.mean(fullChainGain)
subChainGain_Mean = np.mean(subChainGain)



I = np.eye(8)
I2 = np.eye(2)
Loss = []
Gain = []
for i in range(40000):
    ArrayA = xTest[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)
    i1, i2, j1, j2 =  GetIndexFrom(y_PreRes_np_Res[i])
    LetNet_sub = mat(np.zeros((2, 2)), dtype=float)
    LetNet_sub[0, 0] = ArrayA[i1, j1]
    LetNet_sub[0, 1] = ArrayA[i1, j2]
    LetNet_sub[1, 0] = ArrayA[i2, j1]
    LetNet_sub[1, 1] = ArrayA[i2, j2]

    Full_Gain = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    Sub_Gain = math.sqrt(1 / 2) * np.linalg.norm(LetNet_sub, ord='fro')
    Gain.append(Sub_Gain)
    Loss.append(Full_Gain - Sub_Gain)

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
print("LetNet(200000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道增益均值", fullChainGain_Mean)
print("测试样本的穷举法子信道增益均值", subChainGain_Mean)
print("穷举法损失均值", fullChainGain_Mean - subChainGain_Mean)
print('预测准确率', accuracy)
print('预测子信道增益均值', Gain_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)



