# lstm(Long Short-Term Memory models) :
#  연속적인 데이터를 몇개씩 잘라서 작업을 할 것인가?
#  lstm은 연속된(일정한 구간안에 있는) 데이터를 가지고 다음을 예측하는 모델이다.
#  시(time)계열 계산에 최적화된 로직
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

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

# x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
# y = array([4,5,6,7])
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape((x.shape[0], 1, 1))   # (4,3) -> (4,3,1)
print("x.shape : ", x.shape)   # (4, 3, 1)
# print(x)

#2. 모델구성
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(1,1)))
model.add(Dense(100))                                     
model.add(Dense(50))
model.add(Dense(3))
model.summary()

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=30, mode='auto')
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, callbacks=[early_stopping])

x_input = array(1164.5)
x_input = x_input.reshape((1,1,1))

yhat = model.predict(x_input)
print(yhat)

