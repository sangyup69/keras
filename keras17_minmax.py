# Data pre-processing
# 정규화 
#    MinMaxScalar : 어떤 부류의 데이터(1~60000)를 0~1 의 범위로 scaling    https://rfriend.tistory.com/270
#                   X_MinMax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# 정규화의 장점 : 처리속도가 빨라진다.
#    x 값을 전처리 할때, y값은 전처리 할 필요 없다.
#    다만 x_input 값을 전처리 할 경우, x 값을 전처리할 때 같이 하고 이후에 데이터를 잘라야 함.
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],[8,9,10],
           [9,10,11], [10,11,12], [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x)   # training 만 시킴
x = scaler.transform(x)  # 전처리 된 값으로 변환하는 과정

x_train = x[:13, :]
x_predict = x[13:, :]
y_train = y[:13]
# y_predict = y[13:]

# print(x)
print("x.shape : ", x_train)
print("y.shape : ", x_predict)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_shape=(3,), activation='relu'))
model.add(Dense(500))                                    
model.add(Dense(100))      # activation의 default는 'linear'                           
model.add(Dense(50))                               
model.add(Dense(1))
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100)

# 1번에서 전처리한 x값(0~1)의 비중과 아래의 x_input 값을 단순 전처리하면 비중이 다름으로 문제가 발생 
# 그러므로, 원래의 x값으로 fit한 함수를 가지고 transform 하면 됨
import numpy as np

x_input = x_predict.reshape((1,3))
print(x_input)

yhat = model.predict(x_input)
print(yhat)

