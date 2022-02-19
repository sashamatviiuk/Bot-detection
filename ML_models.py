from lib import *
from hyper_func import *

class Classifier():
    
    def __init__(self, X_train, y_train, seed, clf, params):
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.clf = clf
        self.params = params

    def classifier(self):
        if self.clf == 'LogisticRegression':
            return LogisticRegression(random_state=self.seed, solver='liblinear')
        if self.clf == 'SGDClassifier':
            return SGDClassifier(loss='hinge', random_state=self.seed)
        if self.clf == 'DecisionTreeClassifier':
            return DecisionTreeClassifier(random_state=self.seed, class_weight='balanced')
        if self.clf == 'KNeighborsClassifier':
            return KNeighborsClassifier()

    def grid_params(self):
        return GridSearchCV(self.classifier(), self.params)

    def fit_classifier(self):
        return self.grid_params().fit(self.X_train, self.y_train)

    def best_params(self):
        return self.fit_classifier().best_params_

    def pred(self, X):
        return self.fit_classifier().predict(X)

    def pred_proba(self, X):
        return self.fit_classifier().predict_proba(X)[:, 1]

    

    