# yolo : http://blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220993971883
# 잘 만들어 놓은 api를 이용하는 기술 cnn 
# https://www.youtube.com/watch?v=NlpS-DhayQA
# https://www.youtube.com/watch?v=Cgxsv1riJhI
# 인공지능 딥러닝 NVIDIA 젯슨 나노(Jetson Nano),   젯슨나노 자율주행,  OpenCV
# cnn(Convolution layer)의 장점은 feature를 잘 추출한다.
# cnn이 데이터의 각부분을 조각조작 잘라서 feature를 추출하는 것에서는 lstm과 유사
from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
model = Sequential()
# model.add(Conv2D(7, (2,2), padding='same', input_shape=(28,28,1)))  
model.add(Conv2D(7, (2,2), input_shape=(28,28,1)))  
# 7 : 첫번째 layer의 output node수
# (2,2) : 2*2 size로 전체데이터를 중복해서 자르겠다는 의미
# input_shape=(가로,세로,feature)
# 하나의 큰 이미지를 28*28 크기로 잘라서 사용하겠다는 의미

# model.add(Conv2D(16,(2,2)))
# model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8,(2,2)))
model.add(Flatten())   # 2차원의 데이터를 1차원 데이터로 만듦 -> Dense layer에 넣기위해
model.add(Dense(10))
model.add(Dense(1))


model.summary()