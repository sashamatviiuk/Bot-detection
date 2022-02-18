from lib import *
from hyper_func import *

class Classifier():
    
    def __init__(self, X_train, y_train, seed):
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed

    def classifier(self, clf):
        if clf == 'LogisticRegression':
            return LogisticRegression(random_state=self.seed, solver='liblinear')
        if clf == 'SGDClassifier':
            return SGDClassifier(loss='hinge', random_state=self.seed)
        if clf == 'DecisionTreeClassifier':
            return DecisionTreeClassifier(random_state=self.seed, class_weight='balanced')
        if clf == 'KNeighborsClassifier':
            return KNeighborsClassifier()

    def pred(self, X):
        return self.classifier().predict(X)

    def pred_proba(self, X):
        return self.classifier().predict_proba(X)[:, 1]

    