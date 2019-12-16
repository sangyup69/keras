# 모델부분을  lstm 으로 수정할것
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
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1))) 
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train,y_train, epochs=10000, batch_size=1, callbacks=[early_stopping])


x2 = np.array([7,8,9,10])
# print(x2)
# print(x2.shape)
x2 = x2.reshape((1, x2.shape[0], 1))  # (4,) -> (1,4,1)
 
# print(x2)
# print(x2.shape)
y_predict = model.predict(x2)
print(y_predict)

# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
# # print('acc :', acc)
# print('mse :', mse)
# print('loss :', loss)