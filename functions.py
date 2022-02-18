import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

my_path = os.path.abspath(__file__ + '/..')

def percent_each_type(df, col1, col2):
    d = {}
    for i in np.unique(df[col1]):
        d[i] = df[df[col1] == i][col2].count() / df.shape[0]
    return d

def graph(df1, df2, cols, events=3, type_graph='distplot', bins=10, figsize=(15,10), save=True, name_fig='graph'):
    nrows, ncols = len(cols), events
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    n = 1
    for col in cols:
        for event in range(1, events+1):
            arr1 = df1[df1.event_type == event]
            arr2 = df2[df2.event_type == event]
            plt.subplot(nrows, ncols, n)
            plt.title('Column name \'{}\', Event Type: {}'.format(col, event))
            if type_graph == 'hist':
                plt.hist(arr1[col], bins=bins, label='Person')
                plt.hist(arr2[col], bins=bins, label='Bot')
                plt.xlabel('Values')
            elif type_graph == 'distplot':
                sns.distplot(np.log(arr1[col]+1e-12), label='Person')
                sns.distplot(np.log(arr2[col]+1e-12), label='Bot')
                plt.xlabel('Values')
                plt.ylabel('Density')
            plt.legend()
            n += 1
    if save:
        plt.savefig(my_path + '/graph/' + name_fig + '_' + type_graph + '.png')

def bootstrap(n, arr, cols, cat_col):
    df1 = pd.DataFrame()
    for i in np.unique(arr[cat_col]):
        dt = arr[arr[cat_col] == i]
        df2 = pd.DataFrame()
        for col in cols:
            z = int(percent_each_type(df=arr, col1=cat_col, col2=col)[i]*n)
            df2[cat_col] = pd.Series([i]*z)
            df2[col] = np.random.choice(dt[col], size=z)
        df1 = df1.append(df2)
    return df1