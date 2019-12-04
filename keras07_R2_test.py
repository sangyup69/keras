from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])            


model = Sequential()
model.add(Dense(1000, input_shape=(1, ), activation='relu'))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=500, batch_size=1)

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
 
# 문제1. R2를 0.5 이하로 줄이시오.(pc를 과부하시키면 정확도가 낮아짐)
#  layer는 input과 output 포함 5개 이상, 노드는 각 layer당 5개 이상
#  batch_size = 1
#  epochs = 100 이상
# ※ training을 많이 시킬수록 좋은 것이 아니고, 어느정도 이후에는 과부하가 걸림으로, 과부하가 걸리는 시점을 찾는 것이 중요


# mse : 115.9632339477539
# loss : 115.96323661804199
# [[5.150244 ]
#  [5.145881 ]
#  [5.142556 ]
#  [5.136174 ]
#  [5.129286 ]
#  [5.1221056]
#  [5.1149244]
#  [5.1097484]
#  [5.1057754]
#  [5.1035485]]
# RMSE :  10.768622739309802
# R2 :  -13.056149782009713



# y = wx + b  : weight, bias
# x, y 는 정제된 데이터로서 사람이 제공하며, 최적의 w를 구하는 것이 목적임.
# w => loss(cost) 를 최소화 하는 것이 중요함

# deep learning : 심층적으로 학습시킴 
#     layer 의 깊이를 tunning 하는 것, hyper parameter를 조정하는 것으로 deep learning을 할 수 있다.

# 회귀모델, 선형회귀 => y = wx + b
# RMSE : mse 값에 루트를 한 값으로 값이 작을수록 좋은 값임(0가 이상적인 값)
# R2 : r2값이 높을 수록 좋은값임.
# RMSE 와 R2 값을 병행하여 사용하는 것이 좋음

# x, y 값으로 training하고 evaluate, predict 함으로, 제대로 된 평가값을 구할 수 없는 문제가 발생함
# 그래서, x,y 값으로 training하고, test 데이터는 별개로 만들어 사용함
# 데이터 사용비율은 training data : test data = 7 : 3 정도

