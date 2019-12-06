#1. 데이터 (2개 이상의 모델을 입력으로 받을 경우, )
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501,601), range(711,811), range(100)])

x2 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
# print(x1.shape) # (100, 3)
# print(y1.shape) # (100, 3)
# print(x2.shape) # (100, 3)
# print(y2.shape) # (100, 3)

from  sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size=0.4, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=66, test_size=0.5, shuffle=False)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size=0.4, shuffle=False)
x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=66, test_size=0.5, shuffle=False)
# print(x2_test.shape) # (20, 3)

# 2. 모델구성(함수형 모델)   
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
middle1 = Dense(3)(dense3)

input2 = Input(shape=(3,))
xx = Dense(5, activation='relu')(input2)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
middle2 = Dense(3)(xx)

# concatenate
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(30)(merge1)
output1 = Dense(15)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

model = Model(inputs=[input1, input2], outputs=[output1, output2])   #####
model.summary()


# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y2_val]))
# 하나의 변수(데이터)가 들어갈 자리에 2개의 변수(데이터)가 들어갈 경우에 리스트형태로 만들어서 넣으면 됨


# 4. 평가예측
mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print('mse :', mse)
# print('mse :', mse[0])
# print('mse :', mse[1])
# print('mse :', mse[2])
# print('mse :', mse[3])
# print('mse :', mse[4])

y_predict = model.predict([x1_test, x2_test])
# y1_predict, y2_predict = model.predict([x1_test, x2_test])
# print(y_predict)


# RMSE 
from  sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE-1 : ", RMSE(y1_test, y_predict[0]))
print("RMSE-2 : ", RMSE(y2_test, y_predict[1]))
print("RMSE : ", (RMSE(y1_test, y_predict[0])+RMSE(y2_test, y_predict[1]))/2)


# R2
from  sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y_predict[0])
r2_y2_predict = r2_score(y2_test, y_predict[1])

print("R2-1 : ", r2_y1_predict)
print("R2-2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict)/2)


# https://hwiyong.tistory.com/100
# https://keras.io/