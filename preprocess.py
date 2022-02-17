from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preparation():
    # Prepare the data
    # df1 - DataFrame for alive data
    # df2 - DataFrame for bot data
    # features - ['value', 'duration']
    # cat_feature - 'event_type'

    def __init__(self, df1, df2, features, cat_feature):
        self.df1 = df1
        self.df2 = df2
        self.features = features
        self.cat_feature = cat_feature
  
    def bootstrap(self):
        return bootstrap(n=self.df1.shape[0], arr=self.df2, cols=self.features, cat_col=self.cat_feature)

    def plot_graph(self, type_graph='distplot', save=True, name_fig='graph'):
        return graph(df1=self.df1, df2=self.df2, cols=self.features, type_graph=type_graph, save=save, name_fig=name_fig)

    def create_df(self, name='class'):
        self.df1[name] = 1
        self.df2[name] = 0
        df = pd.concat([self.df1, self.df2])
        X = df.loc[:, self.features.append(self.cat_feature)]
        y = df[name]
        return df, X, y

    def train_test(self):
        _, X, y = self.create_df()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        return X_train, X_test, y_train, y_test

    def scalar(self):
        sc = StandardScaler()
        X_train, X_test, _, _ = self.train_test()
        X_train_pre = sc.fit_transform(X_train)
        X_test_pre = sc.transform(X_test)
        return X_train_pre, X_test_pre