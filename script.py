from preprocess import *
from ML_models import *
from report import *
import warnings
warnings.filterwarnings('ignore')

random_state = 23

# Read data
df1 = pd.read_csv('data/alive.csv', delimiter=',')
df2 = pd.read_csv('data/bot.csv', delimiter=',')

print(f'Shape of df1 is {df1.shape}')
print(f'Shape of df2 is {df2.shape}')

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

# Split on train test and scale data
X_train_pre, X_test_pre, y_train, y_test = prep2.prep_split_data()

# LogisticRegression
params = {'penalty':['l1','l2'], 'C':[0.01, 0.1, 1, 10, 100, 1000]}
logreg = LogRegClassifier(X_train=X_train_pre, X_test=X_test_pre, y_train=y_train, y_test=y_test, params=params, seed=random_state)

print(f'Best parameters of LogisticRegression model: {logreg.find_best_params()}')

y_pred = logreg.pred()

report = Report(y_pred=y_pred, y_test=y_test)
report.classification_report()

print(report.f1_score())

