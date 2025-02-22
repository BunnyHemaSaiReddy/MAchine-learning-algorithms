from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix,accuracy_score
#from mlxtend.evaluate import plot_decision_regions

data=load_iris().get('data')
target=load_iris()['target']

for i,t in enumerate(load_iris()['target_names']):
    plt.scatter(data[target==i,0],data[target==i,1],label=t)
#plt.show()
#print(data[:9],target[:9],data[target==0,0],data[target==0,1])


def standed_scaler(x):
    'to maintain mean=0 and standed deviation as 1'
    #z=x-mean/std
    z=(x-np.mean(x))/np.std(x)
    return z

#data=standed_scaler(data)

X,x,Y,y=train_test_split(data,target)

'''
def hop_count(y):
    unique=np.unique(y)
    #print('unique',unique)
    zeros=np.zeros((len(y),len(unique)))
    for i,j in enumerate(unique):
        zeros[y==j,i]=1
    #print(zeros)
    return zeros

def softmax(z):
    'e^x/sum(e^x)'
    exp=np.exp(z-np.max(z,axis=1,keepdims=True))
    #print(z,exp)
    return exp/np.sum(exp,axis=1,keepdims=True)

def logg_regg(x_stand,y,iter=1000,learning_ratio=0.01):
    l=len(np.unique(y))
    y=hop_count(y)
    global m,c
    n_samples,n_features=x_stand.shape
    m=np.zeros((n_features,l))
    c=np.zeros((1,l))
    for i in range(iter):
        model=x_stand@m+c
        z=softmax(model)
        dec_slope=(1/n_samples)*(x_stand.T@(z-y))
        dec_coeff=(1/n_samples)*(np.sum(z-y))
        m-=learning_ratio*dec_slope
        c-=learning_ratio*dec_coeff
    return m,c


def predict(x_test):
    pre=x_test@m+c
    z=softmax(pre)
    return np.argmax(z,axis=1)


logg_regg(X,Y)
pre=predict(x)
print(pre)


print("accuracy-sk",accuracy_score(y,pre))
print(np.sum(pre==y)/len(y)*100,"%")


def cm(y,pre):
    #unique
    u=np.unique(y)
    con_mat=np.zeros((len(u),len(u)))
    dict={j:i for i,j in enumerate(u)}
    for i,j in zip(y,pre):
        row=dict[i]
        col=dict[j]
        con_mat[row,col]+=1
    return con_mat
print(confusion_matrix(y,pre))
print(cm(y,pre))
'''
model=LogisticRegression(max_iter=200)
model.fit(X,Y)


plt.figure(figsize=(10,10))
for i,t in enumerate(load_iris()['target_names']):
    plt.scatter(data[target==i,0],data[target==i,1],label=t)
x_graph=np.linspace(data[:,0].min(),data[:,0].max(),100)
y_graph=np.linspace(data[:,1].min(),data[:,1].max(),100)
print(x_graph.shape,y_graph.shape)
x1_graph=np.linspace(data[:,2].min(),data[:,2].max(),100)
y2_graph=np.linspace(data[:,3].min(),data[:,3].max(),100)
xx,yy=np.meshgrid(x_graph,y_graph)
xx1,yy2=np.meshgrid(x1_graph,y2_graph)
print(xx.ravel().shape,yy.ravel().shape,xx1.ravel().shape,yy2.ravel().shape)
p=np.c_[xx.ravel(),yy.ravel(),xx1.ravel(),yy2.ravel()]
print(p.shape,x.shape)
predict=model.predict(p)
predict.resize(xx.shape)
print(xx.shape,yy.shape,predict.shape)
plt.contourf(xx,yy,predict,alpha=0.3,cmap=plt.cm.Paired)
plt.colorbar()
#plt.legend()
plt.show()



