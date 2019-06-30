import VerifyCodeDataset
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
from keras.models import save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras import Sequential

POS = 3

def codes2name(code):
    dict = ['0','1','2','3','4','5','6',
            '7','8','9','a','b','c','d',
            'e','f','g','h','i','j','k',
            'l','m','n','o','p','q','r',
            's','t','u','v','w','x','y',
            'z']
    res = ''

    for c in code:
        res+=dict[int(c)]

    return res

def code2name(code):
    dict = ['0', '1', '2', '3', '4', '5', '6',
            '7', '8', '9', 'a', 'b', 'c', 'd',
            'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y',
            'z']

    return dict[int(code)]

def plot_train_history(train_history, train):
    plt.plot(train_history.history[train])
    plt.title('CNN Captcha '+train.upper())
    plt.ylabel(train.upper())
    plt.xlabel('Epoches')
    plt.legend(['train',train.lower()],loc='upper left')
    plt.show()

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(20,28)
    if num>25: num = 25
    for i in range(0,num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx])
        title = str(POS)+" label="+str(code2name(labels[idx]))
        if len(prediction) > 0:
            title+=",prediction="+str(code2name(prediction[idx]))
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()



# 读取第0位
(x_train,y_train) , (x_test,y_test) = VerifyCodeDataset.load_data(code_length=4,load_pos=POS)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape, x_test)
# print(y_test.shape, y_test)

# 归一化
x_train_normalize = x_train.astype('float32')/255
x_test_normalize = x_test.astype('float32')/255

# One-Hot
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

plot_images_labels_prediction(x_train, y_train, [], 0, 25)

# 网络搭建 训练第n位的识别

model = Sequential(name='CNN captcha verify 1')
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 96, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=6144, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=35, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x_train_normalize, y_train_onehot, epochs=150, batch_size=128, verbose=1)

plot_train_history(train_history, 'acc')

plot_train_history(train_history, 'loss')

scores = model.evaluate(x_test_normalize, y_test_onehot, verbose=1)

print("Loss",scores[0],"Accuracy",scores[1])

prediction = model.predict_classes(x_test_normalize)

plot_images_labels_prediction(x_test,y_test, prediction,0,25)

if scores[1]>0.99:
    print('高于99准确率，导出')
    save_model(model=model,filepath='models/{0}PosRecognize.h5'.format(POS))





