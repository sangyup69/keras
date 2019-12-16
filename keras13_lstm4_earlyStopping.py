# overfitting(과적합) 해결방법  https://blog.naver.com/jinty/221717568493
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],[8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# print(x)
# print("x.shape : ", x.shape)
# print("y.shape : ", y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))   # (4,3) -> (4,3,1)
# print("x.shape : ", x.shape)   # (4, 3, 1)
# print(x)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(3,1))) 
model.add(Dense(1000))                                    
model.add(Dense(500))                               
model.add(Dense(50))   
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')
# model.fit(x, y, epochs=100, verbose=0)
model.fit(x, y, epochs=10000, callbacks=[early_stopping])  # https://tykimos.github.io/2017/07/09/Early_Stopping/

x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)