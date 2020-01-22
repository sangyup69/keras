from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import model_from_json

#1. 데이터
import openpyxl
filename = "c:\Study\Keras\Pro\exchange.xlsx"
book = openpyxl.load_workbook(filename)
sheet = book.worksheets[0]


xs = []
ys = []
for row in sheet.rows:
    xs.append(row[1].value)    
    ys.append([row[2].value, row[3].value, row[4].value])   
       
x = array(xs)
y = array(ys)

x = x.reshape((x.shape[0], 1, 1))   # (4,3) -> (4,3,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(1,1)))  
model.add(Dense(10))                                       
model.add(Dense(5))
model.add(Dense(3))
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1, batch_size=1)

 
# serialize model to JSON
model_json = model.to_json()
with open("model_exchange.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_exchange.h5")
print("Saved model to disk")


'''

# later...


x_input = array(1163.65)
x_input = x_input.reshape((1,1,1))

# load json and create model
json_file = open('model_exchange.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_exchange.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='mse')

yhat = loaded_model.predict(x_input)
yhat = yhat.reshape((3,1))
print(yhat)
print(yhat[1])
'''