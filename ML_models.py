from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score

import optuna
from sklearn.metrics import log_loss

class LogRegClassifier():

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def best_params(self, *params):
        clf = GridSearchCV(self.model(), *params)
        clf.fit(self.X_train, self.y_train)
        return clf.best_params_

    
