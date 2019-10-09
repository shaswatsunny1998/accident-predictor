# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:21:00 2019

@author: Shaswat
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#Reading csv files
data_accident=pd.read_csv('accident.csv')
data_fall=pd.read_csv('fall.csv')

#Doing preprocessing
print(data_accident.info())
print(data_fall.info())
data_fall=data_fall.dropna()

l=np.zeros(np.shape(data_fall)[0])
data_fall["result"]=l
l=np.ones(np.shape(data_accident)[0])
data_accident["result"]=l


XY_fall=data_fall.iloc[:,2:4].values
XY_accident=data_accident.iloc[:,1:3].values

result_fall=data_fall.iloc[:,-1].values
result_acc=data_accident.iloc[:,-1].values

XY_input=np.concatenate((XY_fall,XY_accident),axis=0)
result=np.concatenate((result_fall,result_acc),axis=0)

sc=StandardScaler()
XY_input=sc.fit_transform(XY_input)



#Training Model and testing model
X_train,X_test,y_train,y_test=train_test_split(XY_input,result,test_size=0.3)
dic={}



#Training Logistic Regression
Logic=LogisticRegression()
parameters={"penalty":['l2'],'C':[1.0,0.5,2.0,0.7],'solver':('newton-cg','lbfgs', 'liblinear', 'sag', 'saga')}
grid_Logistic=GridSearchCV(Logic,parameters,cv=5,scoring='accuracy',n_jobs=-1)
gd_logic=grid_Logistic.fit(X_train,y_train)
Logic_final=gd_logic.best_estimator_
Logic_final.fit(X_train,y_train)
cm=confusion_matrix(y_true=y_test,y_pred=Logic_final.predict(X_test))
acc=accuracy_score(y_test,Logic_final.predict(X_test))
dic["Logistic Regression"]=acc



#Training KNN classifier
knn=KNeighborsClassifier()
parameters_1={'n_neighbors':[4,5,6,7,8,9,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree','kd_tree','brute'],'leaf_size':[30,20,10,40]}
grid_knn=GridSearchCV(knn,parameters_1,cv=5,scoring='accuracy',n_jobs=-1)
gd_knn=grid_knn.fit(X_train,y_train)
knn_final=gd_knn.best_estimator_
knn_final.fit(X_train,y_train)
cm_1=confusion_matrix(y_true=y_test,y_pred=knn_final.predict(X_test))
acc_1=accuracy_score(y_test,knn_final.predict(X_test))
dic["K-Neighbors"]=acc_1


#Traininig SVM classifier
svm=SVC()
parameters_2={'C':[7.0],'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
grid_svm=GridSearchCV(svm,parameters_2,cv=5,scoring='accuracy',n_jobs=-1)
gd_svm=grid_svm.fit(X_train,y_train)
svm_final=gd_svm.best_estimator_
svm_final.fit(X_train,y_train)
cm_2=confusion_matrix(y_true=y_test,y_pred=svm_final.predict(X_test))
acc_2=accuracy_score(y_test,svm_final.predict(X_test))
dic["SVM classifier"]=acc_2

#Training Naive Bayes Theorem
gauss=GaussianNB()
gauss.fit(X_train,y_train)
cm_3=confusion_matrix(y_true=y_test,y_pred=gauss.predict(X_test))
acc_3=accuracy_score(y_test,gauss.predict(X_test))
dic["Naive Bayes"]=acc_3

#Training Decision Tree Classifier
tree=DecisionTreeClassifier()
parameters_3={"criterion":['gini','entropy'],"splitter":['best','random'],}
grid_tree=GridSearchCV(tree,parameters_3,cv=5,scoring='accuracy',n_jobs=-1)
grid_tree.fit(X_train,y_train)
tree_final=grid_tree.best_estimator_
tree_final.fit(X_train,y_train)
cm_4=confusion_matrix(y_true=y_test,y_pred=tree_final.predict(X_test))
acc_4=accuracy_score(y_test,tree_final.predict(X_test))
dic["Decision Tree"]=acc_4



#Training Random Forest Classifier
random=RandomForestClassifier()
parameters_4={'n_estimators':[5,10,15,20,17,25,24,30],'criterion':['gini','entropy']}
grid_random=GridSearchCV(random,parameters_4,cv=5,scoring='accuracy',n_jobs=-1)
grid_random.fit(X_train,y_train)
random_final=grid_random.best_estimator_
random_final.fit(X_train,y_train)
cm_5=confusion_matrix(y_true=y_test,y_pred=random_final.predict(X_test))
acc_5=accuracy_score(y_test,random_final.predict(X_test))
dic["Random Classifier"]=acc_5


#Training XGD boost
model=xgb.XGBClassifier()
model.fit(X_train,y_train)
cm_6=confusion_matrix(y_true=y_test,y_pred=model.predict(X_test))
acc_6=accuracy_score(y_test,model.predict(X_test))
dic["XG Boost"]=acc_6

#Analysing the results of the Results
Estimators=[]
Accuracy=[]
for i in dic:
    Estimators.append(i)
    Accuracy.append(dic[i]*100)
d={'Estimators':Estimators,"Accuracy":Accuracy}
df=pd.DataFrame(data=d)
plt.figure(num=3)
plt.ylim(0,100)
plt.title("All classification estimators with accuracy score")
sns.barplot(x='Estimators',y='Accuracy',data=df)
plt.show()


#Finalizing the model
model=tree_final
print("Enter the Value of X acceleration: ")
x=int(input())
print("Enter the Value of Y acceleration: ")
y=int(input())
f=sc.transform(np.array([[x,y]]))
x=f[0][0]
y=f[0][1]
t=model.predict(np.array([[x,y]]))
if(t==0):
    print("The phone has fallen")
else:
    print("There is a Car accident")


#Bonus Problem
def bonus(XY_input,result,model):
    print("Enter the Value of X acceleration: ")
    x=int(input())
    print("Enter the Value of Y acceleration: ")
    y=int(input())
    print("Enter the situation")
    z=input()
    if("fall" in z):
        res=0
    else:
        res=1
    
    XY_input=np.concatenate((XY_input,np.array([[x,y]])),axis=0)
    result=np.concatenate((result,np.array([res])),axis=0)
    sc=StandardScaler()
    XY_input=sc.fit_transform(XY_input)
    model.fit(XY_input,result)
    t=model.predict(np.array([[x,y]]))
    if(t==0):
        print("The phone has fallen")
    else:
        print("There is a Car accident")
    return XY_input,result

    
    
    
    


