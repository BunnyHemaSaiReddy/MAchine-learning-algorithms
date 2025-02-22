#decision tree

from sklearn.datasets import load_breast_cancer as lbc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

data=lbc()
x=np.array(data['data'])
y=np.array(data['target'])
#print(x,y)
X,x,Y,y=train_test_split(x,y,test_size=0.1,random_state=56)
model=DecisionTreeClassifier()
model.fit(X,Y)
pre=model.predict(x)
print(score(y,pre))

plot_tree(model,max_depth=3,filled=True)
plt.show()
#by mathematically  -- gini index

class Node:
    def __init__(self,gini,predict_class,num_sample,num_sample_class):
        self.threshold=0
        self.num_sample_class=num_sample_class
        self.num_sample=num_sample
        self.value=predict_class
        self.gini=gini
        self.feature_index=0
        self.right=None
        self.left=None
        
class d_tree:
    def __init__(self,max_depth=3):
        self.max_d=max_depth
        #self.tree=None
        
        
    def _gini_impurity(self,y):
        m=len(y)
        return 1.0-sum((np.sum(y==c)/m)**2 for c in np.unique(y))
    
    
    def grow_tree(self,x,y,depth=0):
        num_sample_class=[np.sum(y==i) for i in np.unique(y)]
        predicted_class=np.argmax(num_sample_class)
        
        node=Node(gini=self._gini_impurity(y),num_sample=len(y),num_sample_class=num_sample_class,predict_class=predicted_class)
        #print(node)
        if depth<self.max_d:
         index,thr=self.best_split(x,y)
         if index is not None:
            left_index=x[:,index]<thr
            x_left,y_left=x[left_index],y[left_index]
            x_right,y_right=x[~left_index],y[~left_index]
            node.left=self.grow_tree(x_left,y_left,depth+1)
            node.right=self.grow_tree(x_right,y_right,depth+1)
            node.threshold=thr
            node.feature_index=index
        return node
    
    def best_split(self,x,y):
        m,n=x.shape
        if m<=1:
            return None,None
        index=None
        thresholds=None
        best_gini=1.0-sum((np.sum(y==i)/m)**2 for i in np.unique(y))
        no_sample_class=[np.sum(y==i) for i in np.unique(y)]
        for idx in range(n):
            thr,classes=zip(*sorted(zip(x[:,idx],y)))
            n_left=[0]*len(np.unique(y))
            n_right=no_sample_class.copy()
            for k in range(1,m):
                c=classes[k-1]
                n_left[c]+=1
                n_right[c]-=1
                gini_left=1.0-sum((n_left[j]/k)**2 for j in np.unique(y))
                gini_right=1.0-sum((n_right[j]/(m-k))**2 for j in np.unique(y))
                gini=(k*gini_left+(m-k)*gini_right)/m
                if thr[k]==thr[k-1]:
                    continue
                if gini<best_gini:
                    best_gini=gini
                    thresholds=(thr[k]+thr[k-1])/2
                    index=idx   
        return index,thresholds
    
    
    def fit(self,x,y):
        self.tree=self.grow_tree(x,y)
        
        
    def predict(self,test):
        return [self._predict(self.tree,i) for i in test]
    
    
    def _predict(self,node,i):
        if node.left is None or node.right is None:
          return node.value
        if i[node.feature_index]<node.threshold:
            return self._predict(node.left,i)
        else:
            return self._predict(node.right,i)
        
        
   
model=d_tree(max_depth=3)
model.fit(X,Y)
pre=model.predict(x)
print(score(y,pre))