import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import sklearn.cluster as c


data=ds.load_iris()
data=data.data
#print(data)
model=c.k_means(data,3)
#print(model[0])
def intilizing_centroid(data,k):
    np.random.seed(42)
    index=np.random.choice(data.shape[0],k,replace=True)
    #return data[index]
    return [[2,10],[5,8],[1,2]]


def assign_cluster(data,centroid):
    clus=[]
    for _ in data:
        norm=np.linalg.norm(_-centroid,axis=1)
        arg=np.argmin(norm)
        clus.append(arg)
    print("clus----",clus)
    return np.array(clus)


def update_centroid(data,clus,k):
    new=np.zeros((k,data.shape[1]))
    for _ in range(k):
        new[_]=np.mean(data[clus==_],axis=0)
    print('------------------',new)
    return new

def kmeans(data,k,iterations=5,tolerance=1e-4):
    centroid=intilizing_centroid(data,k)
    for _ in range(iterations):
        cluster=assign_cluster(data,centroid)
        new_centroid=update_centroid(data,cluster,k)
        if (np.all(abs(new_centroid-centroid)<tolerance)):
            print(_)
            break
        centroid=new_centroid
    return cluster,centroid

data=np.array([
    [2,10],
    [2,5],
    [8,4],
    [5,8],
    [7,5],
    [6,4],
    [1,2],
    [4,9]
])
model=kmeans(data,3)
print(model[1])
# plt.scatter(x=data[:,0],y=data[:,1],c=model[0])
# plt.scatter(x=model[1][:,0],y=model[1][0:,1],marker='x',s=200,c='red')
# plt.show()
#print(model)