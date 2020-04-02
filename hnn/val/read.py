from keras.models import *

from keras.utils import *
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os
path="C:\\Users\\Hoyoun Lee\\Dropbox\\대학원\\1-1\\영상처리\\project\\keras\\val\\"
y=[]
x=[]
for file in os.listdir(path):
    if file.endswith(".txt"): 
        x1 = []
        x2 = []
        f = open(path+file, 'r')
        tmp=f.readline()[:-1]
        tmp=list(map(int, tmp))
        y.append(tmp)
        tmp=f.readline()[:-1].split(' ')[:-1]
        tmp=list(map(int, tmp))
        x1.append([tmp])
        tmp=f.readline()[:-1].split(' ')[:-1]
        tmp=list(map(int, tmp))
        x1.append([tmp])
        tmp=f.readline()[:-1].split(' ')[:-1]
        tmp=list(map(int, tmp))
        x1.append([tmp])
        f.readline()
        f.readline()
        tmp=f.readline()[:-1].split(' ')[:-1]
        tmp=list(map(int, tmp))
        x2.append([tmp])
        tmp=f.readline()[:-1].split(' ')[:-1]
        tmp=list(map(int, tmp))
        x2.append([tmp])
        tmp=f.readline()[:-1].split(' ')[:-1]
        tmp=list(map(int, tmp))
        x2.append([tmp])

        x1 = np.array(x1)
        x1 = list(x1.T)
        x2 = np.array(x2)
        x2 = list(x2.T)
        x.append([x1,x2])
        f.close()
x=np.array(x)
y=np.array(y)
x = x.reshape(50,2,256,3)
x.shape
print(x.shape)
# x = x.reshape(401,6,256,1)
print(x.shape)
print(y.shape)
x = x.astype('float32')
x /= 255

# xhat_idx = np.random.choice(x.shape[0], 20)
# xhat = x[xhat_idx]
# print(xhat.shape)

y = np_utils.to_categorical(y)
cnn = load_model('model_final.h5')
score = cnn.evaluate(x, y, verbose=0)
print(cnn.metrics_names)
print(score)