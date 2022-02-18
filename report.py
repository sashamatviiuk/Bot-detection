import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

my_path = os.path.abspath(__file__ + '/..')

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

    def test_overfitting(self, n_round, metric_name='roc_auc'):
        metric_train, metric_test = self.roc_auc_score()
        abs_delta = np.abs(metric_train - metric_test)
        rel_delta = np.abs(metric_train - metric_test) / metric_train

        metrics = pd.Series()
        metrics['Название метрики'] = metric_name
        metrics['Метрика на train'] = np.round(metric_train, n_round)
        metrics['Метрика на test'] = np.round(metric_test, n_round)
        metrics['Абсолютное изменение'] = np.round(abs_delta, n_round)
        metrics['Относительное изменение'] = np.round(rel_delta, n_round)
        return metrics

    def plot_overfitting(self, model, save=True, metric_name='roc_auc', name_fig='overfitting'):
        # model -> str
        metric_train, metric_test = self.roc_auc_score()
        fig = plt.figure(dpi=80)
        plt.bar(['train', 'test'], [metric_train, metric_test])
        plt.title('Переобучение')
        plt.ylabel(metric_name)
        plt.savefig(my_path + '/graph/' + name_fig + '_' + model + '.png')
        plt.close()
        




    