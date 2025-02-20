from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

data = load_iris()
X = data.data
y= data.target

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

joblib.dump(knn,'knnmodel.pkl')

