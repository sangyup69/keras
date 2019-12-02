from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])  # train 과 test 를 분리(train과 test 값을 달리 해야 좀더 정확하게 테스트할 수 있음.)
y_test = np.array([11,12,13,14,15,16,17,18,19,20])  # 기계가 training한 값을 테스트할 때 동일하게 사용하면 데이터 정제 효과가 있음
x_predict = np.array([21,22,23,24,25])              # train : test = 7 : 3 (데이터 비율)


model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))  # input_dim=1  데이터 칼럼이 하나.
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

y_predict = model.predict(x_predict)
print(y_predict)