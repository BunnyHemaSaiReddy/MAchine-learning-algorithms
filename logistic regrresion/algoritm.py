from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.datasets import load_diabetes,load_iris
import pandas as pd
import seaborn 


data=load_iris().get('data')
target=load_iris()['target']
print(len(data),len(target))

t=np.transpose(data)
 
target_names= {'setosa':0, 'versicolor':1, 'virginica':2}
column_in_data=['sepal length','sepal width','petal length','petal width']

dict={'sepal length':t[0],'sepal width':t[1],'petal length':t[2],'petal width':t[3],'target':target}
final_data=pd.DataFrame(dict)
print(final_data.head())

#seaborn.heatmap(final_data.corr())
#plt.show()

plt.scatter(t[0],target)
plt.scatter(t[1],target,color='green')
plt.scatter(t[2],target,color='black')
plt.scatter(t[3],target,color='red')
plt.show()

'''cla=StandardScaler()
cla.fit_transform(data)
cla.transform(data)'''
 

X,x,Y,y=train_test_split(data,target)

def logistic_fit(x,y):
    row,col=x.shape
    global m,c
    m=np.zeros(col)
    c=0
    for i in range(10000):
        z=x@m+c
        sig=1/(1+np.exp(-z))
        #gradient
        dm=(1/row)*(x.T@(z-y))
        dc=(1/row)*np.sum(z-y)
        
        m-=0.01*dm
        c-=0.01*dc
    
def predict(x):
    y=x@m+c
    #sigmoid
    sig=1/(1+np.exp(-y))
    return [2 if i >0.86 else 1 if  i>0.66 else  0 for i in sig ]
'''model=LogisticRegression()
model.fit(X,Y)
pre=model.predict(x)
print(model.score(x,y))'''
logistic_fit(X,Y)
pre= predict(x)
print("accuracy",accuracy_score(y,pre))
print(confusion_matrix(y,pre),pre,sep='\n')
#print(model.predict([[1,5,10,3]]))
