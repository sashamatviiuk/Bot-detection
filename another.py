def graph(df1, df2, cols):

        graph(df1, df2, cols=['value', 'duration'], type_graph='distplot', save=True, name_fig='data_graph')
        graph(df1, df2, cols=['value', 'duration'], type_graph='hist', save=True, name_fig='data_graph')

        graph(df1, df2, cols=['value', 'duration'], type_graph='distplot', save=True, name_fig='bootstrap_graph')
        graph(df1, df2, cols=['value', 'duration'], type_graph='hist', save=True, name_fig='bootstrap_graph')

        
        
        #bootstrap_bot['class'] = 0
        #person_df['class'] = 1

        #df = pd.concat([person_df, bootstrap_bot])

        #X = df.loc[:, ['event_type', 'value', 'duration']]
        #y = df['class']

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        #scaler = StandardScaler()

        #X_train_pre = scaler.fit_transform(X_train)
        #X_test_pre = scaler.transform(X_test)

        #return X_train_pre, X_test_pre, y_train, y_test