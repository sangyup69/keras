from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])            


model = Sequential()
# model.add(Dense(100, input_dim=1, activation='relu'))    
model.add(Dense(100, input_shape=(1, ), activation='relu')) 
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# 선형회귀(y=wx+b), 분류 
# 분류모델에서는 accuracy를 사용, 회귀모델에서는 mse를 사용
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=100, batch_size=1)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
# print('acc :', acc)
print('mse :', mse)
print('loss :', loss)


y_predict = model.predict(x_test)
print(y_predict)

# rmse 
from  sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# R2(1에 가까울수록 좋고, 0에 가까울수록 않좋다를 지표로만 사용)
from  sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


# R2 값을 높이면, RMSE값은 낮아진다.
 
#  mse : 0.8816581964492798
# loss : 0.8816581562161445
# [[10.576825]
#  [11.473668]
#  [12.369848]
#  [13.265856]
#  [14.161862]
#  [15.05787 ]
#  [15.953879]
#  [16.849886]
#  [17.745893]
#  [18.641901]]
# RMSE :  0.9389670778646177
# R2 :  0.893132221416531

# mse : 7.361268039574043e-09
# loss : 7.361268217209727e-09
# [[10.999893]
#  [11.999898]
#  [12.999903]
#  [13.999908]
#  [14.999913]
#  [15.999917]
#  [16.999922]
#  [17.99993 ]
#  [18.999931]
#  [19.999939]]
# RMSE :  8.580048347509828e-05
# R2 :  0.9999999991076699