import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

class Report():
    def __init__(self, y_train, y_test, y_pred_train, y_pred_test):
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test

    def classification_report(self, output_dict=True, name='logreg_report', dir='results'):
        base_report = classification_report(self.y_test, self.y_pred_test, output_dict=output_dict)
        base_report_df = pd.DataFrame(base_report).transpose()
        base_report_df.to_csv(dir + '/' + name, sep='\t', encoding='utf-8')
        return base_report_df

    def f1_score(self, average='binary'):
        return f1_score(self.y_test, self.y_pred_test, average=average)

    def roc_auc_score(self):
        metric_test = roc_auc_score(self.y_test, self.y_pred_test)
        metric_train = roc_auc_score(self.y_train, self.y_pred_train)
        return metric_train, metric_test



    