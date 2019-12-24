# lstm(Long Short-Term Memory models) :
#  연속적인 데이터를 몇개씩 잘라서 작업을 할 것인가?
#  lstm은 연속된(일정한 구간안에 있는) 데이터를 가지고 다음을 예측하는 모델이다.
#  시(time)계열 계산에 최적화된 로직
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

#1. 데이터
import openpyxl
filename = "c:\Temp\exchange.xlsx"
book = openpyxl.load_workbook(filename)
sheet = book.worksheets[0]

xs = []
ys = []
for row in sheet.rows:
    ys.append(row[1].value)    
    xs.append([row[2].value, row[3].value, row[4].value])   
       
x = array(ys)
y = array(xs)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape((x.shape[0], 1, 1))   # LSTM 모델일 경우 사용
print("x.shape : ", x.shape)  
# print(x)

#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_shape=(1, ), activation='relu'))   # DNN
model.add(LSTM(1, activation='relu', input_shape=(1,1)))    # LSTM
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(3))
model.summary()

#3. 실행
early_stopping = EarlyStopping(monitor='acc', patience=30, mode='auto')
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=10, batch_size=1, callbacks=[early_stopping])

x_input = array(1163.9)  # [1.1086 1.2940 109.34]
x_input = x_input.reshape((1,1,1))   # LSTM 모델일 경우 사용
# x_input = x_input.reshape((1, 1))   # DNN 모델일 경우 사용

yhat = model.predict(x_input)
print(yhat)

# [[  1.1777534   1.180474  106.01982  ]]