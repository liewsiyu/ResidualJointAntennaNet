# 这是运行结果截图的程序,最初的一版，一直没改过
# 因为文章中使用的关键数据   accuracy + 预测的mean capacity + 运行时间部分的代码没啥毛病, 就没重新再改重新跑了
# 所以截图只用了上述三个数据

# 结果打印看起来很多，其实屁用没有
# print("基于信道容量, 信噪比:", a)
# print("KNN(200000个测试样本)")
# print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
# print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
# print("测试样本的全信道容量均值", fullChainCapacity_Mean)
# print("测试样本的穷举法子信道容量均值", subChainCapacity_Mean)
# print("穷举法损失均值", fullChainCapacity_Mean - subChainCapacity_Mean)
# print('预测准确率', accuracy)
# print('预测子信道容量均值', Predict_Capacity_Mean)
# print('预测损失均值', Loss_Mean)
# print('预测损失方差', Loss_Variance)



import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from time import time
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
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv')).iloc[0:200000, 1:]
label_array = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\容量标签(p=10行列).csv')).iloc[0:200000, 1:]


#分类器
knn = KNeighborsClassifier()
kf = KFold(n_splits=5,shuffle=True)


Loss_Capacity = []          #5次 容量损失列表(全信道容量-预测子信道容量)   输出结果求平均
KNN_Score = []              #5次 准确率列表  输出结果求平均
Predict_Capacity = []       #10w(5折*20000)预测子信道容量列表   输出结果求平均
trainTime = []              #5次 训练时间列表 输出结果求平均
testTime = []               #5次 预测时间列表 输出结果求平均

I1 = np.eye(8)
I2 = np.eye(2)

for train_index, test_index in kf.split(dataset):
    xTrain, xTest = dataset.iloc[train_index], dataset.iloc[test_index]
    yTrain, yTest = label_array.iloc[train_index], label_array.iloc[test_index]

    trainStart = time()             # 训练模型 + 计算训练时间
    knn.fit(xTrain,yTrain)
    train = time() - trainStart
    trainTime.append(train)
    print('trainTime', trainTime)

    testStart = time()
    predict = knn.predict(xTest)     # 预测模型 + 计算预测时间
    test = time() - testStart
    testTime.append(test)
    print('testTime', testTime)

    yPredict_np = np.array(predict)      #预测的y值
    yTrue_np = np.array(yTest)           #实际的y值
    xTest_np = np.array(xTest)           #实际的x值

    same = 0
    for i in range(0, len(yTrue_np)):
        if yTrue_np[i] == yPredict_np[i]:
            same = same + 1
    KNN_Score.append(same / len(yTrue_np))
    print(KNN_Score)

    #我们用的是10000个数据5折，所以每折就20000个
    for i in range(40000):
        ArrayA = xTest_np[i].reshape(8, 8)
        ArrayA = np.matrix(ArrayA)

        KNN_i1, KNN_i2, KNN_j1, KNN_j2 = GetIndexFrom(yPredict_np[i])    # 反推i,j
        KNN_Sub = mat(np.zeros((2, 2)), dtype=float)
        KNN_Sub[0, 0] = ArrayA[KNN_i1, KNN_j1]
        KNN_Sub[0, 1] = ArrayA[KNN_i1, KNN_j2]
        KNN_Sub[1, 0] = ArrayA[KNN_i2, KNN_j1]
        KNN_Sub[1, 1] = ArrayA[KNN_i2, KNN_j2]
        Full_Capacity  =  np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
        Sub_Capacity  = np.log2(np.linalg.det(I2 + a * KNN_Sub.T * KNN_Sub / 2))

        Predict_Capacity.append(Sub_Capacity)
        Loss_Capacity.append(Full_Capacity  - Sub_Capacity)
    print('Loss_Capacity: ',np.mean(Loss_Capacity))


accuracy = np.mean(KNN_Score)
Predict_Capacity_Mean = np.mean(Predict_Capacity)
Loss_Mean = np.mean(Loss_Capacity)
Loss_Variance = np.var(Loss_Capacity)


train = np.mean(trainTime)
test = np.mean(testTime)


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
fullChainCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道容量(p=10行列).csv").iloc[:, 1:]
fullChainCapacity= fullChainCapacity[0:200000]
fullChainCapacity= np.asarray(fullChainCapacity, np.float32)
subChainCapacity = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道容量(p=10行列).csv").iloc[:, 1:]
subChainCapacity = subChainCapacity[0:200000]
subChainCapacity = np.asarray(subChainCapacity, np.float32)
# 全信道容量文章里面没用到
fullChainCapacity_Mean = np.mean(fullChainCapacity)
# ES算法的最佳子信道容量 获取最优ES_Capacity.py来得到
subChainCapacity_Mean = np.mean(subChainCapacity)
# -------------------------------------------------------------------------------------------


#fullChainCapacity_Mean  subChainCapacity_Mean有上面那块提到的问题, 下面使用这两个变量计算出来的结果都有问题
#但是有问题的结果没用上, 你可以选择忽略 or  重新整理成ResNet容量(返修版本那样重新跑)
#或者你可以看看xlsx文件, 我把有问题的行删了，但是剩下来的部分可以对应截图文件看

print("基于信道容量, 信噪比:", a)
print("KNN(200000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))


#没用到
print("测试样本的全信道容量均值", fullChainCapacity_Mean)
#用另外的程序重新更新了   获取最优ES_Capacity.py
print("测试样本的穷举法子信道容量均值", subChainCapacity_Mean)
#没用到
print("穷举法损失均值", fullChainCapacity_Mean - subChainCapacity_Mean)


print('预测准确率', accuracy)
print('预测子信道容量均值', Predict_Capacity_Mean)



#没用到
print('预测损失均值', Loss_Mean)
#没用到
print('预测损失方差', Loss_Variance)
