from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import * 
import matplotlib.pyplot as plt
import numpy as np
load = load_digits()
X = load.data
Y = load.target
print(X.shape, Y.shape)
class PCa:
    def __init__(self, n_components):
        self.n_comp = n_components
    def fit(self, X, Y=None):
        self.X_ = np.mean(X, axis=0)
        X = X - self.X_
        cov = np.cov(X, rowvar=False)
        e_values, e_vector = np.linalg.eigh(cov)
        sort = np.argsort(e_values)[::-1]
        self.components = e_vector[:, sort[:self.n_comp]]
    def transform(self, X):
        x = X - self.X_
        return np.dot(x, self.components)
pca = PCa(49)
pca.fit(X)
X_pca = pca.transform(X)
print("Transformed data shape:", X_pca.shape)
plt.matshow(X_pca[1].reshape(7,7))
plt.show()