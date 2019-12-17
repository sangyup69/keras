# 이미 만들어진 모델중에서 잘된것들을 재사용하기 위해 사용

#1. 데이터
import numpy as np

x = np.array(range(1,101))  # 1-100
y = np.array(range(1,101))
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
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(100, input_shape=(1, ), activation='relu'))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

model.save('./save/savetest01.h5')
print('Good job!!!')

'''
# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

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
'''