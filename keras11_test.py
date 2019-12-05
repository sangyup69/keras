# R2 0.5 이하로 만들기(PC에 과부하를 발생시키면....)
# 조건 : hidden layer는 5개이상
#        node는 10개 이상
#        epochs 100개이상
#        batch_size = 1

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(101,201)])  
y = np.array([range(201,301)])
print(x.shape)   # (2,100)
x = np.transpose(x)   
y = np.transpose(y)   
print(x.shape)   # (100,2)


from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False)

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(100, input_shape=(2,), activation='relu')) 
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

# model.summary()


# 3. training
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train,y_train, epochs=100, batch_size=1)
model.fit(x_train,y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc :', acc)

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


