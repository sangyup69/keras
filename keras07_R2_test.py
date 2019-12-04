from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])            


model = Sequential()
model.add(Dense(500, input_shape=(1, ), activation='relu'))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=1000, batch_size=1)

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
 
# 문제1. R2를 0.5 이하로 줄이시오.(pc를 과부하시키면 정확도가 낮아짐)
#  layer는 input과 output 포함 5개 이상, 노드는 각 layer당 5개 이상
#  batch_size = 1
#  epochs = 100 이상