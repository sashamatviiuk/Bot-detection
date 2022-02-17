import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, f1_score

class Report():
    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test

    def clas_report(self, output_dict=True, name='logreg_report', dir='results'):
        base_report = classification_report(self.y_test, self.y_pred, output_dict=output_dict)
        base_report_df = pd.DataFrame(base_report).transpose()
        base_report_df.to_csv(dir + '/' + name, sep='\t', encoding='utf-8')
