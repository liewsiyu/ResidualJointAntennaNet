import pandas as pd
list = []
count = 1
for i1 in range(0, 10):
    for i2 in range(i1 + 1,10):
            for j1 in range(0, 10):
                for j2 in range(j1 + 1, 10):
                    new = [int(count),int(i1),int(i2),int(j1),int(j2)]
                    list.append(new)
                    count = count + 1
print(list)
listData = pd.DataFrame(list)
listData.to_csv(r'D:\天线选择 郭志斌\正确数据集\返修101022快速查询数据表.csv')
