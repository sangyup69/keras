# ensemble 할때 각각의 input/output data의 shape도 같아야 하지만, 행의 갯수(데이터의 양)도 같아야 한다.
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1 = x[:10, :]
x2 = x[10:, :]
y1 = y[:10]
y2 = y[10:]
# print(x1)
# print("x1.shape : ", x1.shape)
# print("x2.shape : ", x2.shape)
# print("y1.shape : ", y1.shape)
# print("y2.shape : ", y2.shape)
x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))   # (10,3) -> (10,3,1)
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))   # (3,3) -> (3,3,1)
# print("x1.shape : ", x1.shape)   # (10, 3, 1)
# print("x2.shape : ", x2.shape)   # (3, 3, 1)
# print(x1)
# print(x2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size=0.2, shuffle=False)
# x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=66, test_size=0.5, shuffle=False)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size=0.2, shuffle=False)
# x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=66, test_size=0.5, shuffle=False)
# print(x2_test.shape) # (20, 3)

# 2. 모델구성(함수형 모델)   
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,1))
lstm1 = LSTM(100, activation='relu', input_shape=(3,1))(input1)
dense2 = Dense(3)(lstm1)
dense3 = Dense(4)(dense2)
middle1 = Dense(1)(dense3)

input2 = Input(shape=(3,1))
xx = LSTM(100, activation='relu', input_shape=(3,1))(input2)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
middle2 = Dense(1)(xx)

# concatenate
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(30)(merge1)
output1 = Dense(15)(output1)
output1 = Dense(1)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(1)(output2)

model = Model(inputs=[input1, input2], outputs=[output1, output2])   #####
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')
# model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y2_val]), callbacks=[early_stopping])
print(x1_train)
print(x1_train.shape)
print(x2_train)
print(x2_train.shape)
print(y1_train)
print(y1_train.shape)
print(y2_train)
print(y2_train.shape)

model.fit([x1_train, x1_train], [y1_train, y1_train], epochs=100, batch_size=1, callbacks=[early_stopping])


# 4. 평가예측
mse = model.evaluate([x1_test, x1_test], [y1_test, y1_test], batch_size=1)
print('mse :', mse)

y_predict = model.predict([x1_test, x1_test])
print(y_predict)
 
 
'''
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)


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
'''