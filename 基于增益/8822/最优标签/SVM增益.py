import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from numpy import mat
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold
from time import  time


#快开数据函数  得到对应行列坐标
x = pd.read_csv(open(r'D:\天线选择 郭志斌\天线选择数据集\快速查询数据表.csv'), header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]


m = n = 8
I = np.eye(n)
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\全信道矩阵(p=10行列).csv')).iloc[0:200000, 1:]
label_array = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\增益标签(p=10行列).csv')).iloc[0:200000, 1:]

a = 10
#分类器
svm = SVC()
kf = KFold(n_splits=5,shuffle=True)


Loss_Gain = []          #5次 增益损失列表(全信道增益-预测子信道增益)   输出结果求平均
SVM_Score = []          #5次 准确率列表  输出结果求平均
Predict_Gain = []       #10w(5折*20000)预测子信道增益列表   输出结果求平均
trainTime = []              #5次 训练时间列表 输出结果求平均
testTime = []               #5次 预测时间列表 输出结果求平均

for train_index, test_index in kf.split(dataset):
    xTrain, xTest = dataset.iloc[train_index], dataset.iloc[test_index]
    yTrain, yTest = label_array.iloc[train_index], label_array.iloc[test_index]
    #训练模型

    trainStart = time()
    svm.fit(xTrain, yTrain)  # 训练模型
    train = time() - trainStart
    trainTime.append(train)
    print('trainTime', trainTime)

    #进行预测，并记时
    testStart = time()
    yPredict = svm.predict(xTest)  # 进行预测，并记时
    test = time() - testStart
    testTime.append(test)
    print('testTime', testTime)

    #计算准确率
    yTrue_np = np.array(yTest)          #实际的y值
    yPredict_np = np.array(yPredict)    #预测的y值
    print('实际y值:',yTrue_np)
    print('预测y值:', yPredict)
    same = 0
    for i in range(0, len(yTrue_np)):
        if (yTrue_np[i] == yPredict_np[i]):
            same = same + 1
    SVM_Score.append(same / len(yTrue_np) )
    print(SVM_Score)

    xTest_np = np.array(xTest)         #实际的x值
    #我们用的是20000个数据5折，所以每折就40000个
    for i in range(40000):
        G = []
        ArrayA = xTest_np[i].reshape(8, 8)
        ArrayA = np.matrix(ArrayA)

        SVM_i1, SVM_i2, SVM_j1, SVM_j2 = GetIndexFrom(yPredict_np[i])    # 反推i,j
        SVM_Sub = mat(np.zeros((2, 2)), dtype=float)
        SVM_Sub[0, 0] = ArrayA[SVM_i1, SVM_j1]
        SVM_Sub[0, 1] = ArrayA[SVM_i1, SVM_j2]
        SVM_Sub[1, 0] = ArrayA[SVM_i2, SVM_j1]
        SVM_Sub[1, 1] = ArrayA[SVM_i2, SVM_j2]
        Full_Gain = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
        Sub_Gain = math.sqrt(1 / 2) * np.linalg.norm(SVM_Sub, ord='fro')

        Predict_Gain.append(Sub_Gain)
        G.append(Full_Gain - Sub_Gain)
    Loss_Gain.append(np.mean(G))
    print('信道增益损失: ',Loss_Gain)

accuracy = np.mean(SVM_Score)
Gain_Mean = np.mean(Predict_Gain)
Loss_Mean = np.mean(Loss_Gain)
Loss_Variance = np.var(Loss_Gain)


train = np.mean(trainTime)
test = np.mean(testTime)
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




fullChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\全信道增益(p=10行列).csv").iloc[:, 1:]
fullChainGain = fullChainGain[0:200000]
fullChainGain = np.asarray(fullChainGain, np.float32)

subChainGain = pd.read_csv(r"D:\天线选择 郭志斌\正确数据集\子信道增益(p=10行列).csv").iloc[:, 1:]
subChainGain = subChainGain[0:200000]
subChainGain = np.asarray(subChainGain, np.float32)

fullChainGain_Mean = np.mean(fullChainGain)
subChainGain_Mean = np.mean(subChainGain)






print("基于信道增益")
print("SVM(200000个样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))
print("测试样本的全信道增益均值", fullChainGain_Mean)
print("测试样本的穷举法子信道增益均值", subChainGain_Mean)
print("穷举法损失均值", fullChainGain_Mean - subChainGain_Mean)
print('预测准确率', accuracy)
print('预测子信道增益均值', Gain_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)
