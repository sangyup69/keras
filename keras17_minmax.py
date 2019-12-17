# Data pre-processing
# 정규화 
#    MinMaxScalar : 어떤 부류의 데이터(1~60000)를 0~1 의 범위로 scaling    https://rfriend.tistory.com/270
#                   X_MinMax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# 정규화의 장점 : 처리속도가 빨라진다.
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],[8,9,10],
           [9,10,11], [10,11,12], [20000,30000,40000], [30000,40000,50000], [40000,50000,60000]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)   # training 만 시킴
x = scaler.transform(x)  # 실제 값으로 변환하는 과정
print(x)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

# x = x.reshape((x.shape[0], x.shape[1], 1))   # (4,3) -> (4,3,1)
# print(x)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(3,), activation='relu'))
model.add(Dense(5000))                                    
model.add(Dense(1000))                                
model.add(Dense(500))                               
model.add(Dense(50))   
model.add(Dense(1))
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100)

x_input = array([25,35,45])
x_input = x_input.reshape((1,3))
print(x_input)
# print(x_input.shape)

yhat = model.predict(x_input)
print(yhat)


