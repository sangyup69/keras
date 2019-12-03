from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])            


model = Sequential()
# model.add(Dense(100, input_dim=1, activation='relu'))    
model.add(Dense(100, input_shape=(1, ), activation='relu')) 
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100, batch_size=1)

loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc :', acc)
print('loss :', loss)

y_predict = model.predict(x_test)
print(y_predict)

# rmse : mse에 root를 씌워서 loss값을 작게 한 평가지표
from  sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))