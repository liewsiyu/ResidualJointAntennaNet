import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import mat
import math
#快开数据函数  得到对应行列坐标
x = pd.read_csv(open(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv'), header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]

a = 10
m = n = 8
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv')).iloc[:200000, 1:]
label_array = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\增益标签(p=10行列).csv')).iloc[:200000, 1:]
xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)

xTest_np = np.array(xTest)        #实际的x值
Loss_Capacity = []                #5次 容量损失列表(全信道容量-预测子信道容量)   输出结果求平均
Predict_Capacity = []             #10w(5折*20000)预测子信道增益列表   输出结果求平均

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
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)
    i1, i2, j1, j2 = GetIndexFrom(random.randint(1, 784))
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

print("基于信道增益")
print("随机选择算法40000个测试样本)")
print("测试样本的全信道增益均值", fullChainGain_Mean)
print("测试样本的穷举法子信道增益均值", subChainGain_Mean)
print("穷举法损失均值",fullChainGain_Mean- subChainGain_Mean)
print('预测子信道增益均值',Gain_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)