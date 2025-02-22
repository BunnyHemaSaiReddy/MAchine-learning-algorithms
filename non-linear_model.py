from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


x=np.arange(1,101).reshape(-1,1)
y=5*x**2+3*x+10+np.random.randint(1,20000,size=(100,1))

plt.scatter(x,y,s=5,color='green',marker='s')
#plt.show()


'''poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)'''

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
print(xtrain)
degree=2
x_poly=[[j[0]**i for i in range(degree+1)]for j in xtrain]
print(len(x_poly),len(x_poly[0]))



def polyeq(x_poly,y):
    '''a=x.shape[0]
    power=np.ones((a,a))
    k=-1
    for j in x:
        k+=1 
        for i in range(degree+1):
            power[k][i]=j[0]**i
    print(power)'''
    #x_poly*coeff=y
    # to get the square matrix x_poly is multiply by x_poly transpose both sides
    
    # x_poly*coeff*x_poly^T =x_poly^T * y
    
    x_poly_transpose=np.transpose(x_poly)
    coeff=np.linalg.inv(x_poly_transpose@x_poly)@x_poly_transpose@y
    return coeff

coeff=polyeq(x_poly,ytrain)
print(coeff)
def eq(degree,coeff,x):
 y=0
 for j in range(degree+1):    
    y+=coeff[j]*(x**j)
 return y

l=[]
for j in x:
    l.append(eq(2,coeff,j))
plt.plot(x,l)
plt.show()
    
    
    
'''model=LinearRegression()
model.fit(xtrain,ytrain)
p=model.predict(xtest)'''

p=[]
for j in xtest:
    p.append(eq(2,coeff,j))
print(r2_score(p,ytest))