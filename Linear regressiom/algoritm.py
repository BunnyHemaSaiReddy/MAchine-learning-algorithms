from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import random

#y=3x+2
#np.random.seed(0)
#x=np.random.randn(100,1)
x=np.random.randint(0,2000,size=(200,1))
#y=3*x+2+np.random.randn(200,1)
y=3*x+2+np.random.randint(0,5000,size=(200,1))
#print(x,y)
plt.scatter(x,y,color='orange',marker='*',s=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graph of the line : y=mx+c')

def c():
    #y=mx+c
    return np.mean(y)-m*np.mean(x)
x1=np.mean(x)*np.sum(x)
x2=np.sum(x*x)
y1=np.mean(y)*np.sum(x)
y2=np.sum(y*x)
m=(y2-y1)/(x2-x1)
print(m,c())

l=[]
for i in x:
    l.append(i*m+c())


plt.plot(x,l)
plt.show()
def traintest(x,y,size):
    l=list(enumerate(x))
    arr=np.arange(len(x))
    random.shuffle(arr)
    per=size*len(x)
    x_train,x_test,y_train,y_test=[],[],[],[]
    cou=0
    for i in arr:
        cou+=1
        if cou > per:
         x_train.append(l[i][1])
         y_train.append(y[l[i][0]])
         continue
        x_test.append(l[i][1])
        y_test.append(y[l[i][0]])
    return x_train,x_test,y_train,y_test
    
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
x_train,x_test,y_train,y_test=traintest(x,y,0.25)

def lr_fit(x,y):
    x=np.array(x)
    y=np.array(y)
    x1=np.mean(x)*np.sum(x)
    x2=np.sum(x*x)
    y1=np.mean(y)*np.sum(x)
    y2=np.sum(y*x)
    global m_,c_
    m_=(y2-y1)/(x2-x1)
    c_=np.mean(y)-m*np.mean(x)
    #return m_,c_
def lr_predict(X):
    global predict
    predict=[m_*i+c_ for i in X]
    return predict
def coef():
    return predict[1]-predict[0]

model=LinearRegression()
model.fit(x_train,y_train)
pre=model.predict(x_test)
l=model.predict([[2],[3],[4]])
print(l,'\t\t',[l[i]-l[i-1]for i in range(1,len(l))])

print("--",model.coef_)

'''lr_fit(x_train,y_train)
pre=lr_predict(x_test)

'''

print(r2_score(y_test,pre)*100,"% accuracy")

actual_pred=np.sum(np.array([(i-j)**2 for i,j in zip(y_test,pre)]))
actual_meanactual=np.sum(np.array([(i-np.mean(y_test))**2 for i in y_test]))
r2=1-(actual_pred/actual_meanactual)
print(r2)

print(mean_squared_error(y_test,pre))
#error=1/n * E(y-~y)^2
print((1/len(pre))*np.sum(np.array([(i-j)**2 for i,j in zip(y_test,pre)])))

