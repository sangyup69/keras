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

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(1, ), activation='relu'))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
# model.summary()


# 3. training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# import keras
# tb_hist = keras.callbacks.TensorBoard(log_dir='graph', 
#                                       histogram_freq=0, 
#                                       write_graph=True, 
#                                       write_images=True)
from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
tb_hist = TensorBoard(log_dir='graph', 
                      histogram_freq=0, 
                      write_graph=True, 
                      write_images=True)
model.fit(x_train,y_train, 
          epochs=10, 
          batch_size=1, 
          validation_data=(x_val, y_val), 
          callbacks=[early_stopping, tb_hist])

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

# TensorBoard : summary를 보기좋게 함.
# 1. DOS 창에서 C:\Study\Keras>tensorboard --logdir=./graph  실행
# 2. crome 에서 http://localhost:6006/ 을 실행