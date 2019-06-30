import os

import numpy as np

def gen_label(length=4):
    imgs = os.listdir('./image')
    for i in range(length):
        if os.path.exists('{0}.txt'.format(i)):
            print('Already Have {0}.txt'.format(i))
            continue
        with open('{0}.txt'.format(i),'a') as f:
            for t in imgs:
                code = name2code(t.split('.')[0].lower()[i])
                print(code, file=f)


def name2code(name):
    '''
    默认认为name的值域为小写字母和数字
    将文件名变成编码方便进行one_hot

    e.g.
    0:0
    1:1
    2:2
    ...
    a:10
    b:11
    ...
    :return:编码后的字符串
    '''
    res = ''
    for c in name:
        if ord(c)<=ord('9'):
            res+=c
            res+=' '
        elif ord(c) <= ord('z'):
            res+=str(ord(c)-87)
            res+=' '

    return res

def code2name(code):
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


gen_label(4)

# images = os.listdir('./image')
# Y0 = np.loadtxt('0.txt',dtype=np.uint8)
# Y1 = np.loadtxt('1.txt',dtype=np.uint8)
# Y2 = np.loadtxt('2.txt',dtype=np.uint8)
# Y3 = np.loadtxt('3.txt',dtype=np.uint8)
# print(images[0])
# print(Y0[0],Y1[0],Y2[0],Y3[0])

# print(name)
# print(name2code(name))
# with open('label.txt','a') as f:
#     for i in images:
#         name = i.split('.')[0].lower()
#         code = name2code(name)
#         print(code,file=f)

# Y = np.loadtxt('label.txt',dtype=np.uint8)
# print(Y.shape)
# print(len(images))
# # print(Y)
#
# # 验证无误
# for i in range(len(images)):
#     if code2name(Y[i])==images[i].split('.')[0].lower():
#         print(code2name(Y[i]), images[i].split('.')[0].lower(),'correct')
#     else:
#         print(code2name(Y[i]), images[i].split('.')[0].lower(), 'wrong')



