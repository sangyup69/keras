# 모델링할 경우 lstm을 2~3개 이상 사용하면 효율이 저하 될수 있음
# return sequence는 2개이상의 lstm을 사용할 경우 사용함.
# lstm에서 output shape의 dimension을 하나 늘려주기 위해 return_sequences=True 옵션을 사용한다.
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],[8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape((x.shape[0], x.shape[1], 1))   # (10,3) -> (10,3,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(3,1), return_sequences=True)) # lstm에서 output shape의 dimension을 하나 늘려주기 위해 return_sequences=True 옵션을 사용한다.
model.add(LSTM(200, activation='relu', return_sequences=True)) 
model.add(LSTM(300, activation='relu', return_sequences=True)) 
model.add(LSTM(400, activation='relu', return_sequences=True)) 
model.add(LSTM(10)) 
model.add(Dense(1000))                                    
model.add(Dense(500))                               
model.add(Dense(50))   
model.add(Dense(1))
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# model.fit(x, y, epochs=100, verbose=0)
model.fit(x, y, epochs=10000, callbacks=[early_stopping])  # https://tykimos.github.io/2017/07/09/Early_Stopping/

x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_1 (LSTM)                (None, 3, 100)            40800      # lstm에서 output shape의 dimension을 하나 늘려주기 위해 return_sequences=True 옵션을 사용한다.
# _________________________________________________________________
# lstm_2 (LSTM)                (None, 3, 200)            240800
# _________________________________________________________________
# lstm_3 (LSTM)                (None, 3, 300)            601200
# _________________________________________________________________
# lstm_4 (LSTM)                (None, 3, 400)            1121600
# _________________________________________________________________
# lstm_5 (LSTM)                (None, 10)                16440
# _________________________________________________________________
# dense_1 (Dense)              (None, 1000)              11000
# _________________________________________________________________
# dense_2 (Dense)              (None, 500)               500500
# _________________________________________________________________
# dense_3 (Dense)              (None, 50)                25050
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 51
# =================================================================

# hobby : data pre-processing
# speciality : hyper parameter tunning
# http://physics2.mju.ac.kr/juhapruwp/?p=1517     ANN, DNN, CNN, LSTM, RNN, GRUs
