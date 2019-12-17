# layer 단계(modeling 단계)에서 overfitting을 피하는 방법
# node나 layer가 많다고 좋은 결과를 얻을수 없다.
# 오히려 node를 적당히 줄일 경우, 좋은 예측치를 얻을수도 있다.
#1. 데이터
import numpy as np

x = np.array(range(1,101))  # 1-100
y = np.array(range(1,101))
print(x)

'''
x_train = x[:60]  # 데이터 정제 작업
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]  # 1-60
y_val = y[60:80]  # 61-80
y_test = y[80:]   # 81-100
'''
from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False)
# random_state=66 : 데이터를 자르기 전에 random하게 섞어 줌. 난수는 같은 값으로 주는 게 좋음
# shuffle 
# train : val : test = 6 : 2 : 2

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(1000, input_shape=(1, ), activation='relu'))
model.add(Dropout(0.2))   # 현재의 node(1000개) 중에서 20%를 사용하지 않겠다는 의미
model.add(Dense(1000))
model.add(Dropout(0.3))
model.add(Dense(1000))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Dropout(0.7))
model.add(Dense(1000))
model.add(Dropout(0.9))
model.add(Dense(1))

# model.summary()


# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit(x_train,y_train, epochs=10, batch_size=1, validation_data=(x_val, y_val))

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