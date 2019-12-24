# print('aaaaa')

# import tensorflow
# import keras

# print('hellow!!')


# import openpyxl
# filename = "c:\Temp\exchange.xlsx"
# book = openpyxl.load_workbook(filename)
# sheet = book.worksheets[0]
# data = []
# data1 = []
# data2 = []
# for row in sheet.rows:
#     # data.append([
#     #     row[0].value,
#     #     row[1].value,
#     #     row[2].value,
#     #     row[3].value,
#     #     row[4].value
#     # ])
#     data1.append(row[1].value)    
#     data2.append([row[2].value, row[3].value, row[4].value])    
# # print(data)
# for a in data2:
#     print(a)
# #    print(a[0], a[1], a[2], a[3], a[4])/




from numpy import array
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
