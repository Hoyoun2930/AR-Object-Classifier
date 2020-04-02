from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os

path="C:\\Users\\Hoyoun Lee\\Dropbox\\대학원\\1-1\\영상처리\\project\\keras\\histo\\"
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

x = x.reshape(443,2,256,3)
x.shape

print(x.shape)
# x = x.reshape(401,6,256,1)
print(x.shape)
print(y.shape)
x = x.astype('float32')
x /= 255
y = np_utils.to_categorical(y)

idx=np.arange(443)
np.random.shuffle(idx)
# idx1=idx[:444]
# idx2=idx[444:]
# x_train=x[idx1]
# x_test=x[idx2]
# y_train=y[idx1]
# y_test=y[idx2]



from keras import optimizers

Nadam = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.003)

cnn = Sequential()
cnn.add(Convolution2D(64, 2, 3,border_mode="valid",activation="elu",input_shape=(2, 256, 3)))
cnn.add(Dropout(0.5))
cnn.add(Convolution2D(32, 1, 1, border_mode="valid", activation="elu"))
cnn.add(Dropout(0.5))
# cnn.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="th"))

# cnn.add(Convolution2D(64, 1, 3, border_mode="same", activation="elu"))
# cnn.add(Convolution2D(64, 1, 3, border_mode="same", activation="elu"))
# cnn.add(Convolution2D(64, 1, 3, border_mode="same", activation="elu"))
# cnn.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="th"))   
#cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(8, activation="elu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation="softmax"))

cnn.summary()

cnn.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])

cnn.fit(x, y, batch_size=4, nb_epoch=50, verbose=1)

proba=cnn.predict_proba(x)
#print(proba*100)

from keras.models import *

from keras.utils import *

cnn.save('model_final.h5')

#score = cnn.evaluate(x_test, y_test, verbose=0)
#print(cnn.metrics_names)
#print(score)