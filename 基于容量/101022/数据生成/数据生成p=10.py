import math
import pandas as pd
import time
import numpy as np


。
It = np.eye(2)
p =  10
Nt = 2
def maxChannelCapacity(A):
    C_new = 0
    Count_new = 0
    Count = 0
    # 行  i1,i2
    for i1 in range(0, 10):
        for i2 in range(i1 + 1, 10):
            #列 j1,j2
            for j1 in range(0, 10):
                for j2 in range(j1 + 1, 10):
                    B = A[[i1, i2]][:, [j1, j2]]
                    # 根据西电学报的公式      C是B的信道容量
                    C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
                    # 用count来标记标签
                    Count = Count + 1
                    if C > C_new:  # 求最大信道容量
                        C_new = C
                        Count_new = Count
    # 返回C_new是最大信道容量  Count_new是选出的信道标签
    return [C_new, Count_new]




# 组合之后的最大信道增益(行列2*2),标签
def maxChannelGain(A):
    G_new = 0       # 用于更新最优的等效信道增益
    Count_new = 0   # 用于更新最优的标签
    Count = 0
    for i1 in range(0, 10):
        for i2 in range(i1 + 1, 10):
            # 列 i1,i2,i3,i4
            for j1 in range(0, 10):
                for j2 in range(j1 + 1, 10):
                    B = A[[i1, i2]][:, [j1, j2]]
                    G = math.sqrt(1/2) * np.linalg.norm(B, ord='fro')
                    # 用count来标记标签
                    Count = Count + 1
                    if G > G_new:
                        G_new = G
                        Count_new = Count
    # 返回C_new是最大信道增益(那个子矩阵的)  Count_new是选出的信道标签
    return [G_new, Count_new]

matrix_List= []                       # 全信道矩阵
fullCapacity_List = []                # 全信道容量
subCapacity_List = []                 # 子信道容量
fullGain_List = []                    # 全信道容量
subGain_List = []                     # 子信道容量
capacityLabel_List = []               # 容量标签
gainLabel_List = []                   # 增益标签


m = n = 10
length = 200000
I = np.eye(10)
#生成20000个信道样本
for i in range(0,length):
    A = math.sqrt(1.0/2) * (np.random.rand(m,n)+ 1j*np.random.rand(m,n))
    #将复数矩阵映射为实矩阵
    A = np.matrix(abs(A))

    #数据归一化
    nor1 = np.full(A.shape,np.max(A) - np.min(A))          #np.full填充函数(数组的形状，数组中填充的常数)
    A1 = A - np.full(A.shape,np.min(A))
    #divide(m,n)是m/n
    A1 = np.divide(A1,nor1)

    # matrixList是全信道矩阵
    t = np.array(A1).reshape(1, -1)  # reshape(1,-1)转化成1行,现在的A1是8*8的，你要转化成1*64
    matrix_List.append(t[0])

    # full_ChannelCapacity原全信道容量
    fullCapacity = np.log2(np.linalg.det(I + p * A1.T * A1 / n))
    fullCapacity_List.append(fullCapacity)
    # fullGain原全信道增益
    fullGain = math.sqrt(1 / 2) * np.linalg.norm(A1, ord='fro')
    fullGain_List.append(fullGain)

    # 行列组合  R = return[ , ]
    R1 = maxChannelCapacity(A1)
    R2 = maxChannelGain(A1)
    # 行列组合 对应最大信道容量 标签值
    subCapacity = R1[0]
    capacityLabel = R1[1]
    subGain = R2[0]
    gainLabel = R2[1]

    subCapacity_List.append(subCapacity)
    capacityLabel_List.append(capacityLabel)

    subGain_List.append(subGain)
    gainLabel_List.append(gainLabel)

    #算时间
    if i % 1000 == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(i)

# 全信道矩阵
matrixData = pd.DataFrame(matrix_List)
# 全信道容量
fullCapacityData = pd.DataFrame(fullCapacity_List)
# 全信道增益
fullGainData = pd.DataFrame(fullGain_List)
# 子信道容量(穷举法最优2*2子矩阵)
subCapacityData = pd.DataFrame(subCapacity_List)
# 子信道增益(穷举法最优2*2子矩阵)
subGainData = pd.DataFrame(subGain_List)
# 选出的容量信道标签计数
capacityLabelData = pd.DataFrame(capacityLabel_List)
# 选出的容量信道标签计数
gainLabelData = pd.DataFrame(gainLabel_List)


matrixData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\全信道矩阵(p=10行列).csv')
fullCapacityData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\全信道容量(p=10行列).csv')
subCapacityData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\子信道容量(p=10行列).csv')

fullGainData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\全信道增益(p=10行列).csv')
subGainData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\子信道增益(p=10行列).csv')

capacityLabelData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\容量标签(p=10行列).csv')
gainLabelData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022\增益标签(p=10行列).csv')

