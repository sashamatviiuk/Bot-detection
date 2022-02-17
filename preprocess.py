from functions import *
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

class Prep():
    # Prepare the data
    # df1 - DataFrame for alive data
    # df2 - DataFrame for bot data
    # features - ['value', 'duration']
    # cat_feature - ['event_type']

    def __init__(self, df1, df2, features, cat_feature):
        self.df1 = df1
        self.df2 = df2
        self.features = features
        self.cat_feature = cat_feature
  
    def bootstrap(self):
        return bootstrap(n=self.df1.shape[0], arr=self.df2, cols=self.features, cat_col=self.cat_feature)

    def plot_graph(self, type_graph='distplot', save=True, name_fig='graph'):
        return graph(df1=self.df1, df2=self.df2, cols=self.features, type_graph=type_graph, save=save, name_fig=name_fig)
