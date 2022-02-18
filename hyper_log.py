from lib import *

my_path = os.path.abspath(__file__ + '/..')

def Hyper_logreg(X_train, y_train, cv, seed, low, high, step, name_fig='hyperopt_logreg'):
    def objective(params, pipeline, X_train, y_train):
        pipeline.set_params(**params)
        score = cross_val_score(estimator=pipeline, X=X_train, y=y_train, scoring='roc_auc', cv=cv)
        return {'loss': score.mean(), 'params': params, 'status': STATUS_OK, 'score': score.mean()}
 
    def df_results(hp_results):
        results = pd.DataFrame([{**x, **x['params']} for x in  hp_results])
        results.drop(labels=['status', 'params'], axis=1, inplace=True)
        results.sort_values(by=['loss'], ascending=False, inplace=True)
        return results
 
    model_logreg = Pipeline([('lr', LogisticRegression(random_state=seed, solver ='liblinear'))])
 
    search_space = {'lr__penalty' : hp.choice(label='penalty', options=['l1', 'l2']),
                    'lr__C' : hp.quniform(label='C', low=low, high=high, q=step)}
 
    trials_logreg = Trials()
    best_logreg = fmin(fn=partial(objective, pipeline=model_logreg, X_train=X_train, y_train=y_train),
                       space=search_space,
                       algo=tpe.suggest,
                       max_evals=40,
                       trials=trials_logreg,
                       show_progressbar=False)
  
    results_logreg = df_results(trials_logreg.results)

    sns.set_context("talk")
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(x='lr__C', y='score', hue='lr__penalty', data=results_logreg);
    #ax.set_xscale('log')
    ax.set_xlim(low, high)
    #ax.set_ylim(0.6, 0.7)
    ax.grid()
    plt.savefig(my_path + '/graph/' + name_fig + '.png', dpi=300)
    plt.close()