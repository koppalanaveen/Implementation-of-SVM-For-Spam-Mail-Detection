# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import the required packages.
3. Import the dataset to operate on.
4. Split the dataset.
5. Predict the required output.
6. End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KOPPALA NAVEEN
RegisterNumber:  212223100023
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
# DATA.HEAD

![image](https://github.com/user-attachments/assets/abf487df-aa36-4083-8d2a-8b8187af8677)

# DATA.INFO

![image](https://github.com/user-attachments/assets/37a04b0f-1dd4-4a8b-927f-1d066e9b746b)

# DATA.ISNULL

![image](https://github.com/user-attachments/assets/3c5dfd29-29f1-4c6b-8e5b-d0d9f191cef2)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
