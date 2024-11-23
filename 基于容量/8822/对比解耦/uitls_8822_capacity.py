import numpy as np
def decoupledSelection_8822(data,p):
    I = np.eye(8)
    It = np.eye(2)
    Nt = 2
    max_transmit = 0
    T1 = 0
    T2 = 0
    C = 0

    #解耦算法是发射端和接收端分开选择的
    #所以前两个for循环是要把最优的发射T1,T2选出来
    for i in range(0, 8):
        for j in range(i + 1, 8):
            B = data[[i, j], :]
            new_C = np.linalg.det(I + p * B.T * B / Nt)
            if new_C > max_transmit:
                max_transmit = new_C
                T1 = i
                T2 = j

    for i in range(0, 8):
        for j in range(i + 1, 8):
            B = data[[T1, T2]][:, [i, j]]
            new_C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
            if new_C > C:
                C = new_C
    return C


def maxChannelCapacity_8822(A,p):
    C_new = 0
    Count_new = 0
    Count = 0
    Nt = 2
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
                            B = A[[i1, i2]][:, [j1, j2]]
                            #根据西电学报的公式      C是B的信道容量
                            C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
                            #用count来标记标签
                            Count = Count + 1
                            if C > C_new:    #求最大信道容量
                               C_new = C
                               Count_new = Count
    # 返回C_new是最大信道容量  Count_new是选出的信道标签
    return [C_new,Count_new]




def computation_time(test):
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
    return [test,testUnit]

