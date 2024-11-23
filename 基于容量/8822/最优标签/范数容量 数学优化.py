import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import mat
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


xTest_np = np.array(dataset)        #实际的x值
Loss_Capacity = []                #5次 容量损失列表(全信道容量-预测子信道容量)   输出结果求平均
Predict_Capacity = []             #10w(5折*20000)预测子信道增益列表   输出结果求平均

gain_np = np.array(label_array)
I1 =np.eye(8)
I2 =np.eye(2)
for i in range(200):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(gain_np[i])
    Sel_B = mat(np.zeros((2, 2)), dtype=float)
    Sel_B[0, 0] = ArrayA[i1, j1]
    Sel_B[0, 1] = ArrayA[i1, j2]
    Sel_B[1, 0] = ArrayA[i2, j1]
    Sel_B[1, 1] = ArrayA[i2, j2]
    Full_capacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Sub_capacity = np.log2(np.linalg.det(I2 + a * Sel_B.T * Sel_B / 2))
    print(Sub_capacity)
    break
    Predict_Capacity.append(Sub_capacity)
    Loss_Capacity.append(Full_capacity - Sub_capacity)


Predict_Capacity_Mean = np.mean(Predict_Capacity)
Loss_Mean = np.mean(Loss_Capacity)
Loss_Variance = np.var(Loss_Capacity)



fullChainCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道容量(p=10行列).csv").iloc[:, 1:]
fullChainCapacity= fullChainCapacity[0:200000]
fullChainCapacity = np.asarray(fullChainCapacity, np.float32)

subChainCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道容量(p=10行列).csv").iloc[:, 1:]
subChainCapacity = subChainCapacity[0:200000]
subChainCapacity = np.asarray(subChainCapacity, np.float32)

fullChainCapacity_Mean = np.mean(fullChainCapacity)
subChainCapacity_Mean = np.mean(subChainCapacity)






print("基于信道容量，信噪比",a)
print("随机选择算法200000个测试样本)")
print("测试样本的全信道容量均值", fullChainCapacity_Mean)
print("测试样本的穷举法子信道容量均值", subChainCapacity_Mean)
print("穷举法损失均值",fullChainCapacity_Mean- subChainCapacity_Mean)
print('预测子信道容量均值',Predict_Capacity_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)