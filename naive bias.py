import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
iris = load_iris()


X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_priors(y):
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    return dict(zip(classes, priors))

priors = calculate_priors(y_train)
def calculate_statistics(X, y):
    classes = np.unique(y)
    statistics = {}
    for cls in classes:
        X_cls = X[y == cls]
        statistics[cls] = {
            'mean': np.mean(X_cls, axis=0),
            'var': np.var(X_cls, axis=0)
        }
    return statistics

statistics = calculate_statistics(X_train, y_train)
def gaussian_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent
def calculate_class_probabilities(statistics, priors, x):
    probabilities = {}
    for cls, stats in statistics.items():
        probabilities[cls] = np.log(priors[cls])
        for i in range(len(stats['mean'])):
            probabilities[cls] += np.log(gaussian_probability(x[i], stats['mean'][i], stats['var'][i]))
    return probabilities
def predict_single(statistics, priors, x):
    probabilities = calculate_class_probabilities(statistics, priors, x)
    return max(probabilities, key=probabilities.get)
def predict(statistics, priors, X):
    predictions = []
    for x in X:
        predictions.append(predict_single(statistics, priors, x))
    return np.array(predictions)

predictions = predict(statistics, priors, X_test)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


acc = accuracy(y_test, predictions)
print(f'Accuracy: {acc * 100:.2f}%')



model=GaussianNB()
model.fit(X_train, y_train)
pre=model.predict(X_test)
acc = accuracy(y_test, pre)
print(f'Accuracy: {acc * 100:.2f}%')
import pandas as pd
import sweetviz as sw
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

''''# Analyze the dataframe
d = sw.analyze(df)
d.show_html()'''
import dtale
d=dtale.show(df)
fig=d.build_main_url()
print(fig)