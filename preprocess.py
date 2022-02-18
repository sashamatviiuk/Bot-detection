from functions import *
import pandas as pd
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

    def plot_graph(self, type_graph='distplot', save=True, name_fig='graph'):
        return graph(df1=self.df1, df2=self.df2, cols=self.features, type_graph=type_graph, save=save, name_fig=name_fig)

    def create_df(self):
        self.df1['class'] = 1
        self.df2['class'] = 0
        df = pd.concat([self.df1, self.df2], ignore_index=True)
        X = df.loc[:, [self.cat_feature] + self.features]
        y = df['class']
        return X, y

    def prep_split_data(self):
        X, y = self.create_df()
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train_pre = scaler.fit_transform(X_train)
        X_test_pre = scaler.transform(X_test)
        return X_train_pre, X_test_pre, y_train, y_test