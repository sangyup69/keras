#1. 데이터
import numpy as np

x = np.array(range(1,101))  # 1-100
y = np.array(range(1,101))

from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False)

# 2. 모델구성(함수형 모델)
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# input1 = Input(shape=(1,))
# dense1 = Dense(5, activation='relu')(input1)
# dense2 = Dense(3)(dense1)
# dense3 = Dense(4)(dense2)
# dense4 = Dense(5)(dense3)
# dense5 = Dense(5)(dense4)
# dense6 = Dense(5)(dense5)
# dense7 = Dense(5)(dense6)
# dense8 = Dense(100)(dense7)
# dense9 = Dense(5)(dense8)
# dense10 = Dense(5)(dense9)
# dense11 = Dense(5)(dense10)
# output1 = Dense(1)(dense11)

input1 = Input(shape=(1,))
xx = Dense(5, activation='relu')(input1)
xx = Dense(3)(xx)   # 동일한 변수를 사용해도 순차적으로 적용됨으로 문제 없음
xx = Dense(4)(xx)
xx = Dense(5)(xx)
xx = Dense(5)(xx)
xx = Dense(5)(xx)
xx = Dense(5)(xx)
xx = Dense(100)(xx)
xx = Dense(5)(xx)
xx = Dense(5)(xx)
xx = Dense(5)(xx)
output1 = Dense(1)(xx)

model =  Model(inputs=input1, outputs=output1)
model.summary()


# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse :', mse)
print('loss :', loss)


y_predict = model.predict(x_test)
print(y_predict)

# rmse 
from  sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# R2
from  sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


'''
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 1)                 0
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 25
_________________________________________________________________
dense_5 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_6 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_7 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_8 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_9 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_10 (Dense)             (None, 5)                 30
_________________________________________________________________
dense_11 (Dense)             (None, 5)                 30
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 6
=================================================================
Total params: 285
Trainable params: 285
Non-trainable params: 0
_________________________________________________________________
Train on 60 samples, validate on 20 samples
'''