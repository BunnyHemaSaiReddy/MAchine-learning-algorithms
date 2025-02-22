import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset=load_diabetes()
X,Y=dataset.data,dataset.target
#print(X[:5],Y[:5],dataset.feature_names)
y=np.reshape(Y,(len(Y),1))
print(X.shape,y.shape)
samp=np.hstack((X,y))

df=pd.DataFrame(samp,columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6','target'])
'''#sns.heatmap(df.corr(),annot=True)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
j=0
for i in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']:
    ax.scatter(df.get(i),y,zs=j,label=i)
    j+=5
#plt.show()
#plt.legend()

def mc(x,y):
    x=np.array(x)
    y=np.array(y)
    x1=np.mean(x)*np.sum(x)
    x2=np.sum(x*x)
    y1=np.mean(y)*np.sum(x)
    y2=np.sum(y*x)
    m_=(y2-y1)/(x2-x1)
    c_=np.mean(y)-m_*np.mean(x)
    return m_,c_

print(df.target.shape,df.s1.shape,y.shape)

for i in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']:
    #y=df.get('target')
    y=Y
    plt.scatter(df.get(i),y,label=i)
    m,c=mc(df.get(i),y)
    print(m,c)
    plt.plot(df.get(i),df.get(i)*m+c)
    plt.title( i +' vs target')
    #plt.show()

'''

'''df=df.drop(['target','s3','sex','age','s1','s2'],axis=1)
print(df.head())

X=df.loc[:,['bp','s5']]
X=np.array(df.bmi).reshape(-1,1)
print(X[5])'''

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1/3)

model=LinearRegression()
model.fit(x_train,y_train)
pre=model.predict(x_test)
print("score",r2_score(y_test,pre))

def ac(x):
    return np.hstack((x,np.ones((x.shape[0],1))))
def lin_model(x,y):
    #y=mx+c
    
    #adding of c to x
    var=np.hstack((x,np.ones((x.shape[0],1))))
    #findin m --- m=y*x^-1
    
    m=np.linalg.inv(var.T@var)@(var.T@y)
    return m

pre=ac(x_test)@lin_model(x_train,y_train)
    
print("score",r2_score(y_test,pre))