from lib import *

class LogRegClassifier():

    def __init__(self, X_train, X_test, y_train, y_test, params, seed):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.seed = seed
        self.params = params

    def logreg(self):
        return LogisticRegression(random_state=self.seed, solver='liblinear')

    def gridsearch_best_params(self):
        clf = GridSearchCV(self.logreg(), self.params).fit(self.X_train, self.y_train)
        return clf.best_params_

    def logreg_best_params(self):
        return LogisticRegression(random_state=self.seed, solver='liblinear', **self.gridsearch_best_params()).fit(self.X_train, self.y_train)

    def pred(self, X):
        return self.logreg_best_params().predict(X)
    
    def pred_proba(self, X):
        return self.logreg_best_params().predict_proba(X)[:, 1]