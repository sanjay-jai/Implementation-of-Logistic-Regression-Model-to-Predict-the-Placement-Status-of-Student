# Implementation of Logistic Regression Model to Predict the Placement Status of Student

## Aim:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJAI A
RegisterNumber:  212220040142
*/
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
data1= data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test,y_pred) 
accuracy 
from sklearn.metrics import confusion_matrix 
confusion = confusion_matrix(y_test,y_pred) 
confusion
from sklearn.metrics import classification_report 
classification_report1 = classification_report(y_test,y_pred) 
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://user-images.githubusercontent.com/95969295/204101863-2929c227-9f4e-427c-9d12-98b7b63f0da7.png)

![image](https://user-images.githubusercontent.com/95969295/204101885-87252d41-19e4-45fc-bb5f-d3c8b107fe93.png)

![image](https://user-images.githubusercontent.com/95969295/204101913-6959a1dc-f2c6-4423-9736-9c7acee3cc38.png)

![image](https://user-images.githubusercontent.com/95969295/204101933-f247df3d-be17-4aaa-80c0-5d61228201d9.png)

![image](https://user-images.githubusercontent.com/95969295/204101959-0867eb51-c287-4843-a66a-a939d81f95f4.png)

![image](https://user-images.githubusercontent.com/95969295/204101991-963fa46c-11b5-4415-8d4c-a7aa12e6a139.png)

![image](https://user-images.githubusercontent.com/95969295/204102017-4819b8d5-03e0-4d65-8f8d-21f474f5de81.png)

![image](https://user-images.githubusercontent.com/95969295/204102034-adab7b77-5a7e-4226-97a0-7c450633c98c.png)

![image](https://user-images.githubusercontent.com/95969295/204102054-993abbd1-5c47-4251-b092-e4571511b494.png)

![image](https://user-images.githubusercontent.com/95969295/204102073-fd1ccdfd-0ac7-42c5-8a49-8c5e5111ce7a.png)

![image](https://user-images.githubusercontent.com/95969295/204102101-b83ee096-e483-458f-b834-75b19fe259e6.png)

![image](https://user-images.githubusercontent.com/95969295/204102126-e9b91445-a4b5-4791-afce-369bbd335aaa.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
