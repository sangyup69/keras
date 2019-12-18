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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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

# convolution 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))    # Output Shape=(None, 26, 26, 32)
# model.add(Conv2D(32, (4,4), strides=(2,2), input_shape=(10,10,1), activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))     # Output Shape=(None, 24, 24, 64)
model.add(MaxPooling2D(pool_size=2))               # 사소한 변화를 무시해주는 Max Pooling 레이어
model.add(Dropout(0.25))
model.add(Flatten())                               # 영상을 일차원으로 바꿔주는 Flatten 레이어
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))   # 분류모델의 마지막에는  softmax가 필수.  입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # https://crazyj.tistory.com/153

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# 모델 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=30, batch_size=200, verbose=1,
                    callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print('\n Test Accuracy : %.4f' % (model.evaluate(X_test, Y_test)[1])) # [0] : loss,  [1] : accuracy


# _________________________________________________________________    https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320           
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 12, 12, 64)        0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 9216)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               1179776       = (9216 + 1)*128  
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1290
# =================================================================


# 라이젠9 3900X 마티스 RTX2080Ti
# 그래픽카드 ram memory가 큰 것이 좋음
