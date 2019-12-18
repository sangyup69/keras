# X_train(60000, 28, 28) -> x1, x2 각 30000개로 나눌것
# Y_train(60000,) -> y1, y2 각 30000개로 나눌것

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf
import numpy as np

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32') / 255   # (60000, 28, 28, 1)  이미지를 잘라서 분석하기 위해 reshape
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32') / 255      # (10000, 28, 28, 1)
# 255 로 나눈 것은 좀더 좋은 결과를 얻기 위한 데이터 전처리를 위함
# 각 셀의 값을 255로 나누면 0~1 사이의 값으로 변경됨

# 분류모델에서는 compile에서 categorical_crossentropy, 모델링 마지막에는  softmax가 필수
# one hot encoding : 사용하는 부분만 1로 표기하고, 나머지는 0으로 표기하는 것
Y_train = np_utils.to_categorical(Y_train)    # (60000,) -> (60000, 10)   0~9 까지만 인식하면되기 때문에 10개의 행만 있으면 됨
Y_test = np_utils.to_categorical(Y_test)      # (10000,) -> (10000, 10)

from  sklearn.model_selection import train_test_split
x1, x2, y1, y2 = train_test_split(X_train, Y_train, random_state=66, test_size=0.5, shuffle=False)
x1_test, x2_test, y1_test, y2_test = train_test_split(X_test, Y_test, random_state=66, test_size=0.5, shuffle=False)

# ensemble로 convolution(cnn) 신경망의 modeling
input1 = Input(shape=(28,28,1))
dense1 = Conv2D(32, kernel_size=(3,3), activation='relu')(input1)
dense2 = MaxPooling2D(pool_size=2)(dense1)
dense3 = Flatten()(dense2)
dense4 = Dense(10, activation='relu')(dense3)
middle1 = Dense(10)(dense4)

input2 = Input(shape=(28,28,1))
xx = Conv2D(32, kernel_size=(3,3), activation='relu')(input2)
xx = MaxPooling2D(pool_size=2)(xx)
xx = Flatten()(xx)
xx = Dense(10, activation='relu')(xx)
middle2 = Dense(10)(xx)

# concatenate
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(3)(merge1)
output1 = Dense(10, activation='softmax')(output1)

output2 = Dense(3)(merge1)
output2 = Dense(10, activation='softmax')(output2)

model = Model(inputs=[input1, input2], outputs=[output1, output2])   #####
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # https://crazyj.tistory.com/153

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# 모델 실행
history = model.fit([x1,x2], [y1,y2], validation_data=([x1_test, x2_test], [y1_test, y2_test]),
                    epochs=3, batch_size=200, verbose=1,
                    callbacks=[early_stopping_callback])

print('\n loss :%.4f, Accuracy : %.4f' % (model.evaluate([x1_test, x2_test], [y1_test, y2_test]))) # [0] : loss,  [1] : accuracy

# kingkeras@naver.com 윤영선 교수
