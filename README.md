# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
## Step 1:
Import the required packages.
## Step 2:

Read the given dataset and assign x and y array.
## Step 3:

Split x and y into training and test set.
## Step 4:

Scale the x variables.
## Step 5:

Fit the logistic regression for the training set to predict y.
## Step 6:

Create the confusion matrix and find the accuracy score, recall sensitivity and specificity
## Step 7:

Plot the training set results.
## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NITHISHWAR S
RegisterNumber:  212221230071
*/
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df

#assigning x and y and displaying them
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 

#splitting data into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#scaling values and obtaining scaled array of train and test of x
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)

#applying logistic regression to the scaled array
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)

#finding predicted values of y
ypred=c.predict(xtest)
ypred

#calculating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm

#calculating accuracy score
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc

#calculating recall sensitivity and specificity
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec

#displaying regression 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(("pink","purple")))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(("white","violet"))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

```

## Output:
## Dataset:
![image](https://user-images.githubusercontent.com/94164665/173245587-5b65014d-f989-48ab-baff-8a0411ddcae4.png)

## Predicted Y array:
![image](https://user-images.githubusercontent.com/94164665/173245607-24b201d5-d4a0-4f8e-9ce0-8c1729988d41.png)

## Confusion matrix:
![image](https://user-images.githubusercontent.com/94164665/173245626-2c878822-2172-4e91-ad98-60e3ac5abd92.png)

## Accuracy score:
![image](https://user-images.githubusercontent.com/94164665/173245641-eaf0aa83-440d-4455-b465-4e2ea0d591bb.png)

## Recall sensitivity and specificity:
![image](https://user-images.githubusercontent.com/94164665/173245652-9f1994c6-3305-4614-9a75-fcba85367042.png)

## Logistic Regression graph:
![image](https://user-images.githubusercontent.com/94164665/173245674-8e879a2a-33c7-451b-aed6-0d1580fab53b.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

