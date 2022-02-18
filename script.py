from datetime import datetime
start_time = datetime.now()

from preprocess import *
from ML_models import *
from report import *
from hyper_log import *
from lib import *

import warnings
warnings.filterwarnings('ignore')

# Constants
low = 0.1
high = 10
random_state = 23
n_splits = 9
test_size = 0.2
n_round = 4

# Read data
df1 = pd.read_csv('data/alive.csv', delimiter=',')
df2 = pd.read_csv('data/bot.csv', delimiter=',')

#print(f'Shape of df1 is {df1.shape}')
#print(f'Shape of df2 is {df2.shape}')

type_graph = ['distplot', 'hist']
features = ['value', 'duration']
cat_feature = 'event_type'

#print(df1[cat_feature].value_counts(ascending=False))
#print(percent_each_type(df=df1, col1=cat_feature, col2=features[0]))
#print(percent_each_type(df=df2, col1=cat_feature, col2=features[0]))

# Plot data
prep1 = Preparation(df1, df2, features=features, cat_feature=cat_feature)
prep1.plot_graph(type_graph=type_graph[0], save=False, name_fig='raw_data_graph')
prep1.plot_graph(type_graph=type_graph[1], save=False, name_fig='raw_data_graph')

bootstrap_bot = bootstrap(n=df1.shape[0], arr=df2, cols=features, cat_col=cat_feature)
bootstrap_bot = bootstrap(n=df1.shape[0], arr=df2, cols=['value', 'duration'], cat_col='event_type')

prep2 = Preparation(df1=df1, df2=bootstrap_bot, features=features, cat_feature=cat_feature)
prep2.plot_graph(type_graph=type_graph[0], save=False, name_fig='bootstrap_graph')
prep2.plot_graph(type_graph=type_graph[1], save=False, name_fig='bootstrap_graph')

# Create DataFrame
X, y = prep2.create_df()
spl = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

# Split on train test and scale data
X_train_pre, X_test_pre, y_train, y_test = prep2.prep_split_data()

# LogisticRegression
#print('LogisticRegression model')

params = {'penalty':['l1','l2'], 'C':[0.01, 0.1, 1, 10, 100, 1000]}
logreg = LogRegClassifier(X_train=X_train_pre, X_test=X_test_pre, y_train=y_train, y_test=y_test, params=params, seed=random_state)

#print(f'Best parameters: {logreg.gridsearch_best_params()}')

#y_pred_test = logreg.pred(X_test_pre)
#y_pred_train = logreg.pred(X_train_pre)

#report = Report(y_train=y_train, y_test=y_test, y_pred_train=y_pred_train, y_pred_test=y_pred_test)
#classification_report= report.classification_report()

#print('Classification report')
#print(classification_report)

#print('Cross validation metric is ROC-AUC')

model = logreg.logreg_best_params()
#cv_scores = cross_val_score(model, X, y, cv=spl, scoring='roc_auc')
#mean_score, cv_score_std, cv_scores = np.round(np.mean(cv_scores), n_round), np.round(np.std(cv_scores), n_round), np.round(cv_scores, n_round)
#print(f'Cross_val_scores {cv_scores}')
#print(f'Mean cross_val_score is {mean_score}')
#print(f'Std cross_val_score is {cv_score_std}')

#print('ROC-AUC on train test datasets with split 80% / 20%')
#y_pred_proba_test = logreg.pred_proba(X_test_pre)
#y_pred_proba_train = logreg.pred_proba(X_train_pre)
#report = Report(y_train=y_train, y_test=y_test, y_pred_train=y_pred_proba_train, y_pred_test=y_pred_proba_test)

#report.test_overfitting(n_round=n_round)
#overfit_logreg = report.plot_overfitting(model='logreg', save=False)
#roc_pr = report.roc_auc_pr_plot()

hype = Hyper_logreg(X_train=X_train_pre, y_train=y_train, cv=spl, seed=random_state, low=low, high=high, step=1)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))