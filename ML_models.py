from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

import optuna
from sklearn.metrics import log_loss

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

    def find_best_params(self):
        clf = GridSearchCV(self.logreg(), self.params).fit(self.X_train, self.y_train)
        return clf.best_params_

    def logreg_best_params(self):
        return LogisticRegression(random_state=self.seed, solver='liblinear', **self.find_best_params()).fit(self.X_train, self.y_train)

    def pred(self):
        return self.logreg_best_params().predict(self.X_test)