from google.colab import drive
drive.mount('/content/drive')

import numpy as np 
import pandas as pd
from sklearn import svm,metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/test.csv')
train.head()

y_label = train['label'] 
X_data = train.drop('label', axis = 1)

X_train_data = np.array(X_data)/255 
X_test_data = np.array(test)/255

# Tách tập dữ liệu thành tập huấn luyện và tập kiểm tra 
X_train,X_test,y_train,y_test = train_test_split(X_train_data,y_label,test_size=0.2)

# Code dùng bằng SVM
clf = svm.SVC(kernel='linear',C=100)
clf.fit(X_train,y_train)# phân lớp tập (X_train, y_train)
y_pred = clf.predict(X_test)
#print(y_pred)#in dự đoán
print()# so sánh độ chính xác
print('accuracy score = ', metrics.accuracy_score(y_test,y_pred))


print(y_pred)#in dự đoán
