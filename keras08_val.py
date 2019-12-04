
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])

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

# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))
# x_train,y_train 으로 훈련시킬 때,x_val, y_val 값으로 검증해 가면서 훈련시키으로 정도를 향상시킴

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

# https://wdprogrammer.tistory.com/29
# 훈련 데이터에 대한 학습만을 바탕으로 모델의 설정(Hyperparameter)를 튜닝하게 되면 
# 과대적합(overfitting)이 일어날 가능성이 매우 크다. 
# 또한, 테스트 데이터는 학습에서 모델에 간접적으로라도 영향을 미치면 안 되기 때문에 
# 테스트 데이터로 검증을 해서는 안 된다. 
# 그래서 검증(validation) 데이터셋을 따로 두어 매 훈련마다 검증 데이터셋에 대해 평가하여 
# 모델을 튜닝해야 한다.

#  <데이터를 나눌 시 주의점>
# 대표성 : 훈련 데이터셋과 테스트 데이터셋은 전체 데이터에 대한 대표성을 띄고 있어야 한다.
# 시간의 방향 : 과거 데이터로부터 미래 데이터를 예측하고자 할 경우에는 데이터를 섞어서는 안 된다. 
#              이런 문제는 훈련 데이터셋에 있는 데이터보다 테스트 데이터셋의 모든 데이터가 미래의 것이어야 한다.
# 데이터 중복 : 각 훈련, 검증, 테스트 데이터셋에는 데이터 포인트의 중복이 있어서는 안 된다. 
#              데이터가 중복되면 올바른 평가를 할 수 없기 때문이다.

# mse : 5.093170243192224e-12
# loss : 5.093170329928398e-12
# [[11.000001]
#  [12.000002]
#  [13.000003]
#  [14.      ]
#  [15.000003]
#  [16.000002]
#  [17.      ]
#  [18.000002]
#  [19.      ]
#  [20.000002]]
# RMSE :  1.7841612752790171e-06
# R2 :  0.9999999999996142