import os

import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

path = 'image'

# 改成每次读取一位,4位一起需要tensorflow的自定义验证，而keras钟验证是自动执行，或者是我不清楚如何自定义验证方法，所以只能分别训练4个一样的网络，导入
# 不同的数据去训练来实现验证不同的位置
def load_data(code_length = 4, load_pos = 0):
    images = []
    files = os.listdir(path)
    for i in files:
        if i.endswith('.jpg'):
            img = cv.imread(path+'/'+i) # 读入彩色图片
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img = cv.resize(img, (96,32))
            # print(img)
            images.append(img)  # 添加到列表

    X = np.array(images)  # 测试和训练数据一起

    Y = np.loadtxt('{0}.txt'.format(load_pos), dtype=np.uint8)
    # image和label读取完毕

    # 开始分测试集和训练集
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=30)

    return (X_train,Y_train),(X_test,Y_test)



# (x_train,y_train),(x_test,y_test) = load_data(load_pos=1)
#
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

