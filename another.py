from preprocess import *
import warnings
warnings.filterwarnings('ignore')



df1 = pd.read_csv('data/alive.csv', delimiter=',')
df2 = pd.read_csv('data/bot.csv', delimiter=',')

features = ['value', 'duration']
cat_feature='event_type'

example = Prep(df1, df2, features=features, cat_feature=cat_feature)
boot_bot = example.bootstrap()

example2 = Prep(df1=df1, df2=boot_bot, features=features, cat_feature=cat_feature)
example2.plot_graph()