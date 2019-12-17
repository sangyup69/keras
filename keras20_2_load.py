# 이미 만들어진 모델중에서 잘된것들을 재사용하기 위해 사용

#1. 데이터
import numpy as np

x = np.array(range(1,101))  # 1-100
y = np.array(range(1,101))

from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False)

# 2. 모델구성
from keras.models import Sequential, load_model   ####
from keras.layers import Dense
model = Sequential()

model.add(Dense(1, name='dense_0', input_shape=(1, ), activation='relu'))
model.add(load_model('./save/savetest01.h5'))
model.add(Dense(10, name='dense_6'))
model.add(Dense(1, name='dense_7'))

model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_0 (Dense)              (None, 1)                 2
# _________________________________________________________________
# sequential_1 (Sequential)    (None, 1)                 11341
# _________________________________________________________________
# dense_6 (Dense)              (None, 10)                20
# _________________________________________________________________
# dense_7 (Dense)              (None, 1)                 11
# =================================================================


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
