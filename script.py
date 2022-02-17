from preprocess import *
from ML_models import *
from report import *
import warnings
warnings.filterwarnings('ignore')

random_state = 23

# Read data
df1 = pd.read_csv('data/alive.csv', delimiter=',')
df2 = pd.read_csv('data/bot.csv', delimiter=',')

type_graph = ['distplot', 'hist']
features = ['value', 'duration']
cat_feature = 'event_type'

# Plot data
prep1 = Preparation(df1, df2, features=features, cat_feature=cat_feature)
prep1.plot_graph(type_graph=type_graph[0], save=True, name_fig='raw_data_graph')
prep1.plot_graph(type_graph=type_graph[1], save=True, name_fig='raw_data_graph')

bootstrap_bot = prep1.bootstrap()

prep2 = Preparation(df1=df1, df2=bootstrap_bot, features=features, cat_feature=cat_feature)
prep2.plot_graph(type_graph=type_graph[0], save=True, name_fig='bootstrap_graph')
prep2.plot_graph(type_graph=type_graph[1], save=True, name_fig='bootstrap_graph')

# Create DataFrame
df, X, y = prep2.create_df()

# Split on train test
X_train, X_test, y_train, y_test = prep2.train_test()

# Scale data
X_train_pre, X_test_pre = prep2.scale_data()

# LogisticRegression
params = {'penalty':['l2'], 'C':[1]}
logreg = LogRegClassifier(X_train=X_train_pre, X_test=X_test_pre, y_train=y_train, y_test=y_test, params=params, seed=random_state)

y_pred = logreg.pred()

rep = Report(y_pred=y_pred, y_test=y_test)
rep.clas_report()