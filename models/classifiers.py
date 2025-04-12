from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_model(name, X, y):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }
    clf = models[name]
    clf.fit(X, y)
    return clf
