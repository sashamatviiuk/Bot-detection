from preprocess import *
import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv('data/alive.csv', delimiter=',')
df2 = pd.read_csv('data/bot.csv', delimiter=',')

type_graph = ['distplot', 'hist']
features = ['value', 'duration']
cat_feature=['event_type']

example = Preparation(df1, df2, features=features, cat_feature=cat_feature[0])
boot_bot = example.bootstrap()

example2 = Preparation(df1=df1, df2=boot_bot, features=features, cat_feature=cat_feature[0])
example2.plot_graph(type_graph=type_graph[0], save=False, name_fig='graph')

df, X, y = example2.create_df()

