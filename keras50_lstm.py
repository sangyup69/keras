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
    # data.append([
    #     row[0].value,
    #     row[1].value,
    #     row[2].value,
    #     row[3].value,
    #     row[4].value
    # ])
    ys.append(row[1].value)    
    xs.append([row[2].value, row[3].value, row[4].value])   
       
x = array(xs)
y = array(ys)

# x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
# y = array([4,5,6,7])
# print(x)
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)
# x    y
# 123  4
# 234  5
# 345  6
# 456  7

x = x.reshape((x.shape[0], x.shape[1], 1))   # (4,3) -> (4,3,1)
print("x.shape : ", x.shape)   # (4, 3, 1)
# print(x)

#2. 모델구성
model = Sequential()
model.add(LSTM(300, activation='relu', input_shape=(3,1)))   # (행,3,1) input_shape에서 행은 무시
model.add(Dense(100))                                         # 1 : 몇개씩 잘라서 작업할 것인가, 작업을 위한 cut size
model.add(Dense(50))
model.add(Dense(1))
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100)

x_input = array([1.1086,1.294,109.34])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)