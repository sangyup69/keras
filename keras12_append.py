# x1 과 x2 데이터를 합치고, y1 과 y2 데이터를 합친 후, DNN 모델을 완성할 것
#1. 데이터 (2개 이상의 모델을 입력으로 받을 경우, )
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501,601), range(711,811), range(100)])

x2 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

xc = np.concatenate([x1,x2])
yc = np.concatenate([y1,y2])

xc = np.transpose(xc)
yc = np.transpose(yc)
print(xc.shape)  # (100, 6)

from  sklearn.model_selection import train_test_split
xc_train, xc_test, yc_train, yc_test = train_test_split(xc, yc, random_state=66, test_size=0.4, shuffle=False)
xc_val, xc_test, yc_val, yc_test = train_test_split(xc_test, yc_test, random_state=66, test_size=0.5, shuffle=False)

print(xc_test.shape) # (20, 6)


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(100, input_shape=(6, ), activation='relu'))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(6))

# model.summary()


# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit(xc_train,yc_train, epochs=100, batch_size=1, validation_data=(xc_val, yc_val))

loss, mse = model.evaluate(xc_test, yc_test, batch_size=1)
print('mse :', mse)
print('loss :', loss)


y_predict = model.predict(xc_test)
print(y_predict)

# rmse 
from  sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(yc_test, y_predict))

# R2
from  sklearn.metrics import r2_score
r2_y_predict = r2_score(yc_test, y_predict)
print("R2 : ", r2_y_predict)