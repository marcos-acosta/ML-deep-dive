import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/breast_cancer/data.csv')
# Drop last column (not relevant)
data = data.drop([data.columns[-1]], axis=1)

X_features = list(data.columns[2:])
y_features = data.columns[1]
X = data[X_features]
y = data[y_features]
y = y.apply(lambda x: 0 if (x == 'B') else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

nb = GaussianNB()
nb.fit(X_train, y_train)
plot_confusion_matrix(nb, X_test, y_test)
plt.show()