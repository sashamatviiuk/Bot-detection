from lib import *
from ML_models import *

my_path = os.path.abspath(__file__ + '/..')

class Hyper():
    def __init__(self, X_train, y_train, cv, seed, clf, max_evals, model_classifier):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.seed = seed
        self.clf = clf
        self.max_evals = max_evals
        self.model_classifier = model_classifier

    def objective(self, params, pipeline):
        pipeline.set_params(**params)
        score = cross_val_score(estimator=pipeline, X=self.X_train, y=self.y_train, scoring='roc_auc', cv=self.cv)
        return {'loss': score.mean(), 'params': params, 'status': STATUS_OK, 'score': score.mean()}
 
    def df_results(self, hp_results):
        results = pd.DataFrame([{**x, **x['params']} for x in  hp_results])
        results.drop(labels=['status', 'params'], axis=1, inplace=True)
        results.sort_values(by=['loss'], ascending=False, inplace=True)
        return results

    def search(self, name_fig='hyperopt'):
        if self.clf == 'LogisticRegression':
            C_min, C_max, step = 0.1, 10, 0.1

            search_space = {'lr__penalty' : hp.choice(label='penalty', options=['l1', 'l2']),
                            'lr__C' : hp.quniform(label='C', low=C_min, high=C_max, q=step)}

            pipe = Pipeline([('lr', self.model_classifier)])

            trials = Trials()
            best_params = fmin(fn=partial(self.objective, pipeline=pipe),
                               space=search_space,
                               algo=tpe.suggest,
                               max_evals=self.max_evals,
                               trials=trials,
                               show_progressbar=False)
  
            results = self.df_results(trials.results)

            sns.set_context("talk")
            plt.figure(figsize=(8, 8))
            ax = sns.scatterplot(x='lr__C', y='score', hue='lr__penalty', data=results)
            ax.set_xlim(C_min, C_max)
            ax.grid()
            plt.savefig(my_path + '/graph/' + self.clf + name_fig + '.png', dpi=300)
            plt.close()

        elif self.clf == 'SGDClassifier':
            alpha_min, alpha_max, step = 0.00002, 0.00005, 0.000005
            search_space = {'sgd__penalty' : hp.choice(label='penalty', options=['l1', 'l2', 'elasticnet']),
                            'sgd__alpha' : hp.quniform(label='C', low=alpha_min, high=alpha_max, q=step)}

            #search_space = {'C':hp.loguniform("C", np.log(1), np.log(100)),
                            #'kernel':hp.choice('kernel',['rbf','poly']),
                            #'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1))}
            pipe = Pipeline([('sgd', self.model_classifier)])

            trials = Trials()
            best_params = fmin(fn=partial(self.objective, pipeline=pipe),
                               space=search_space,
                               algo=tpe.suggest,
                               max_evals=self.max_evals,
                               trials=trials,
                               show_progressbar=False)
  
            results = self.df_results(trials.results)

            sns.set_context("talk")
            plt.figure(figsize=(8, 8))
            ax = sns.scatterplot(x='sgd__alpha', y='score', hue='sgd__penalty', data=results)
            ax.set_xlim(alpha_min, alpha_max)
            ax.grid()
            plt.savefig(my_path + '/graph/' + self.clf + name_fig + '.png', dpi=300)
            plt.close() 
    
    

    