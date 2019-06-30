from keras import models
import cv2 as cv
import os
import numpy as np
import requests
import random
import matplotlib.pyplot as plt

try:
    model1 = models.load_model('models/0PosRecognize.h5')
    model2 = models.load_model('models/1PosRecognize.h5')
    model3 = models.load_model('models/2PosRecognize.h5')
    model4 = models.load_model('models/3PosRecognize.h5')
    print('Load Successfully!')
except:
    print("Load Unsuccessfully!Please train a new model!")

rand_num = random.randint(0,1000)
base_url = "http://zhjw.scu.edu.cn/img/captcha.jpg?"

intact_url = base_url+str(rand_num)

ret = requests.get(intact_url)
if ret.status_code == 200:
    with open('captcha.jpg','wb') as f:
        for chuck in ret:
            f.write(chuck)

img = cv.imread('captcha.jpg')

img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img = cv.resize(img, (96,32))

# 图片预处理
img_normalize = img.astype('float32')/255

t = []

t.append(img_normalize)

test = np.array(t)

pos0 = model1.predict_classes(test)
pos1 = model2.predict_classes(test)
pos2 = model3.predict_classes(test)
pos3 = model4.predict_classes(test)

def code2name(code):
    dict = ['0', '1', '2', '3', '4', '5', '6',
            '7', '8', '9', 'a', 'b', 'c', 'd',
            'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y',
            'z']

    return dict[int(code)]

res = code2name(*pos0)+code2name(*pos1)+code2name(*pos2)+code2name(*pos3)

import matplotlib.pyplot as plt
def plot_images_prediction(img, res):
    plt.figure('Result Of CNN')
    plt.imshow(img)
    plt.title("Prediction "+res)
    plt.show()

plot_images_prediction(img,res)

# plt.figure()
# plt.imshow(img)
# plt.show()




