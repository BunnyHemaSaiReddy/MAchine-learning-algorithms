import numpy as np
import sklearn.datasets as d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score
data=d.load_digits()
x1,y1=np.array(data['data']),np.array(data['target'])
data=d.load_breast_cancer()
x,y=np.array(data['data']),np.array(data['target'])
#print(x,y)
class knn:
    def __init__(self,k=3) :
        self.k=k
    def euclidiandistance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def fit(self,x,y):
        self.X=x
        self.Y=y
    def _(self,x):
        d=[self.euclidiandistance(x,_) for _ in self.X]
        k_indexs=np.argsort(d)[:self.k]
        k_lables=[self.Y[i] for i in k_indexs]
        return np.bincount(k_lables).argmax()
    def predict(self,x):
        y=[self._(i) for i in x]
        return y

x,X,y,Y=train_test_split(x,y,test_size=0.2,random_state=30)
model=knn(20)
model.fit(x,y)
y_p=model.predict(X)
print(accuracy_score(y_p,Y))
print(confusion_matrix(y_p,Y))
x,X,y,Y=train_test_split(x1,y1,test_size=0.2,random_state=42)
class knn2:
    def __init__(self,k=3) :
        self.k=k
    def euclidiandistance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def fit(self,x,y):
        self.X=x
        self.Y=y
    def _(self,x):
        d=[self.euclidiandistance(x,_) for _ in self.X]
        k_indexs=np.argsort(d)[:self.k]
        k_lables=[self.Y[i] for i in k_indexs]
        return np.mean(k_lables)
    def predict(self,x):
        y=[round(self._(i)) for i in x]
        return np.array(y)
model=KNeighborsRegressor(5)
model.fit(x,y)
p=model.predict(X)
m=knn2(5)
m.fit(x,y)
pr=m.predict(X)
print(r2_score(p,Y))
print(r2_score(Y,pr))