from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# print(X_train[0])
# print(Y_test[0])
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
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
# print(Y_train[:10])
# print(Y_test[:10])

# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2], 1)

# convolution 신경망의 설정
model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(LSTM(3, activation='relu', input_shape=(784,1)))
# model.add(Conv2D(64, (3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=2))      
model.add(Dropout(0.25))
# model.add(Flatten())                      
model.add(Dense(10))
# model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # https://crazyj.tistory.com/153
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# 모델 실행
# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
#                     epochs=30, batch_size=200, verbose=1,
#                     callbacks=[early_stopping_callback])
model.fit(X_train,Y_train, epochs=1, batch_size=1)

# 테스트 정확도 출력
# print('\n Test Accuracy : %.4f' % (model.evaluate(X_test, Y_test)[1])) # [0] : loss,  [1] : accuracy



loss, acc = model.evaluate(X_test, Y_test, batch_size=1)
print('acc :', acc)
print('loss :', loss)


x = x.reshape((x.shape[0], x.shape[1], 1))   # (4,3) -> (4,3,1)
# print("x.shape : ", x.shape)   # (4, 3, 1)
# print(x)
