# x 값은 있어도 y 값이 없는 데이터가 있을 경우, 데이터를 분할하여 사용(예, 시계열 데이터)
# [1 2 3 4 5 6 7 8 9 10]
# x          | y
# 1  2  3  4 | 5
# 2  3  4  5 | 6
# 
# 6  7  8  9 | 10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))   # 정제되지 않은 데이터로 가정

size = 5
def split_x(sdata, size):
    aaa = []
    for i in range(len(sdata)-size+1):
        subset = sdata[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)

x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

print(x_train.shape)
print(x_train)
print(y_train.shape)
print(y_train)

model = Sequential()
model.add(Dense(100, input_shape=(4, ), activation='relu')) 
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100, batch_size=1)


x2 = np.array([7,8,9,10])
# print(x2)
# print(x2.shape)
x2 = x2.reshape((1, x2.shape[0]))  # (4,) -> (1,4)
 
# print(x2)
# print(x2.shape)
y_predict = model.predict(x2)
print(y_predict)

# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
# # print('acc :', acc)
# print('mse :', mse)
# print('loss :', loss)