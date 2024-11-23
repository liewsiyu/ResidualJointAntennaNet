import numpy as np
import pandas as pd
import math


#快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\天线选择 郭志斌\正确数据集\8822\8822快速查询数据表.csv', header=None)
matrix = x.values
def GetLabel(i1, i2, j1, j2):
    for i in range(0, 784):
        if i1 == matrix[i, 1] and i2 == matrix[i, 2] and j1 == matrix[i, 3] and j2 == matrix[i,4]:
            return matrix[i][0]



# 首先从信道矩阵中选出范数最大的元素
def maxNormIndex(A):
    max_Gain = 0
    max_i = 0
    max_j = 0
    for i in range(0, 8):
        for j in range(0, 8):
            # Frobenius范数   每个元素的平方和 再开根号
            G = math.sqrt(A[i,j]*A[i,j])
            if G > max_Gain:
               max_Gain = G
               max_i = i
               max_j = j
    # 返回最大的行标签   列标签
    return [max_i, max_j]

#A Near-Optimal Joint Transmit and Receive Antenna Selection Algorithm for MIMO System
#Blum R S
#MIMO系统联合天线选择算法郎保才
def JTRAS_Capacity_Algorithm_8822(A1, p):
    B = np.mat(np.zeros((2, 2)), dtype=float)
    transmit = {0, 1, 2, 3, 4, 5, 6, 7}
    receive = {0, 1, 2, 3, 4, 5, 6, 7}
    #待选集合中剔除第一对发射和接收天线
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
    #已选集合中增加第一对发射和接收天线
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_C = 0
    add_transmit  = 0
    add_receive = 0
    I2 = np.eye(2)

    for i in transmit:   #行
        for j in receive: #列
            #B[0,0]是从信道矩阵中选出范数最大的元素
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
            #B[0,1]与B[0,0]有相同的行,与B[1,1]有相同的列
            B[0,1] = A1[maxNormIndex(A1)[0],j]
            #B[1,0]与B[0,0]有相同的列,与B[1,1]有相同的行
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            C = np.log2(np.linalg.det(I2 + p * B.T * B / 2))
            if C > max_C:
                max_C = C
                add_transmit = i
                add_receive = j

    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_C,selected_transmit_list[0],selected_transmit_list[1],selected_receive_list[0],selected_receive_list[1]
def JTRAS_Gain_Algorithm_8822(A1):
    transmit = {0, 1, 2, 3, 4, 5, 6, 7}
    receive = {0, 1, 2, 3, 4, 5, 6, 7}
    #待选集合中剔除第一对发射和接收天线
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
    #已选集合中增加第一对发射和接收天线
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_G = 0
    add_transmit = 0
    add_receive = 0
    for i in transmit:   #行
        for j in receive: #列
            B = np.mat(np.zeros((2, 2)), dtype=float)
            #B[0,0]是从信道矩阵中选出范数最大的元素
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
            #B[0,1]与B[0,0]有相同的行,与B[1,1]有相同的列
            B[0,1] = A1[maxNormIndex(A1)[0],j]
            #B[1,0]与B[0,0]有相同的列,与B[1,1]有相同的行
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            G = math.sqrt(1 / 2) * np.linalg.norm(B, ord='fro')
            if G > max_G:
                max_G = G
                add_transmit = i
                add_receive = j

    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_G,selected_transmit_list[0],selected_transmit_list[1],selected_receive_list[0],selected_receive_list[1]





#最优天线选择算法
def maxChannelCapacity(A):
    C_new = 0
    Count_new = 0
    Count = 0
    It = np.eye(2)
    #行  i1,i2
    for i1 in range(0, 8):
        for i2 in range(i1+1, 8):
            #列 j1,j2
            for j1 in range(0, 8):
                for j2 in range(j1+1, 8):
                    #其实对于i1,i2来说 没必要！=   因为i1不可能等于i2，但是这个加保险。
                    if i1 != i2:
                        if j1 != j2:
                            B = np.mat(np.zeros((2, 2)), dtype=float)
                            # 矩阵进行 行列 赋值
                            B[0, 0] = A[i1, j1]
                            B[0, 1] = A[i1, j2]
                            B[1, 0] = A[i2, j1]
                            B[1, 1] = A[i2, j2]
                            #根据西电学报的公式      C是B的信道容量
                            C = np.log2(np.linalg.det(It + p * B.T * B / 2))
                            #用count来标记标签
                            Count = Count + 1
                            if C > C_new:    #求最大信道容量
                               C_new = C
                               Count_new = Count
    # 返回C_new是最大信道容量  Count_new是选出的信道标签
    return [C_new,Count_new]
def maxChannelGain(A):
    G_new = 0.       # 用于更新最优的等效信道增益
    Count_new = 0   # 用于更新最优的标签
    Count = 0
    # 行i1,i2
    for i1 in range(0, 8):
        for i2 in range(i1+1, 8):
            # 列 j1,j2
            for j1 in range(0, 8):
                for j2 in range(j1+1, 8):
                    # 其实对于i1,i2来说 没必要！=   因为i1不可能等于i2，但是这个加保险。
                    if i1 != i2:
                        if j1 != j2:
                            B = np.mat(np.zeros((2, 2)), dtype=float)
                            # 矩阵进行 行列 赋值
                            B[0, 0] = A[i1, j1]
                            B[0, 1] = A[i1, j2]
                            B[1, 0] = A[i2, j1]
                            B[1, 1] = A[i2, j2]
                            # 计算等效信道增益 根号a乘根号b于根号ab.
                            G = math.sqrt(1/2) * np.linalg.norm(B, ord='fro')
                            # 用count来标记标签
                            Count = Count + 1
                            if G > G_new:
                               G_new = G
                               Count_new = Count
    # 返回C_new是最大信道增益(那个子矩阵的)  Count_new是选出的信道标签
    return [G_new, Count_new]




#数据预处理
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\8822\全信道矩阵(p=20行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8)




length = 200000
p = 20  #信噪比


subCapacity_List = []
subGain_List = []

capacityLabel_List = []
gainLabel_List = []


for i in range(0,200000):
    print(i)
    subCapacity,i1_Capacity,i2_Capacity,j1_Capacity,j2_Capacity = JTRAS_Capacity_Algorithm_8822(dataset[i],p)  #第一个返回的子信道容量
    subGain, i1_Gain, i2_Gain, j1_Gain, j2_Gain = JTRAS_Gain_Algorithm_8822(dataset[i])  #第一个返回的子信道增益
                                                                                                  

    subCapacity_List.append(subCapacity)
    subGain_List.append(subGain)
    capacityLabel_List.append(GetLabel(i1_Capacity,i2_Capacity,j1_Capacity,j2_Capacity))
    gainLabel_List.append(GetLabel(i1_Gain, i2_Gain, j1_Gain, j2_Gain))


# 子信道容量(JTRAS最优2*2子矩阵)
subCapacityData = pd.DataFrame(subCapacity_List)
# 选出的容量信道标签计数
capacityLabelData = pd.DataFrame(capacityLabel_List)

# 子信道容量(JTRAS最优2*2子矩阵)
subGainData = pd.DataFrame(subGain_List)
# 选出的容量信道标签计数
gainLabelData = pd.DataFrame(gainLabel_List)


subCapacityData.to_csv(r'D:\天线选择 郭志斌\正确数据集\8822\JTRAS子信道容量(p=20行列).csv')
capacityLabelData.to_csv(r'D:\天线选择 郭志斌\正确数据集\8822\JTRAS容量标签(p=20行列).csv')
subGainData.to_csv(r'D:\天线选择 郭志斌\正确数据集\8822\JTRAS子信道增益(p=20行列).csv')
gainLabelData.to_csv(r'D:\天线选择 郭志斌\正确数据集\8822\JTRAS增益标签(p=20行列).csv')
