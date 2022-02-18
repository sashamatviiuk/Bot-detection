from lib import *

def objective(trial, X, y):
    cv_scores =[]
    
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 500, 1000]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.001, 0.01, 0.1]),
        "num_leaves": trial.suggest_categorical("num_leaves", [3, 5, 10]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 7]),
        "min_split_gain": trial.suggest_categorical("min_split_gain", [0, 0.5]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [1, 5]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 1, 10]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0, 1, 10])}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = lgb.LGBMClassifier(objective="binary", 
                               class_weight='balanced', 
                               random_state=23, 
                               subsample=1.0, 
                               subsample_freq=10, 
                               colsample_bytree=0.85,
                               **param_grid)
    model.fit(X_train,
              y_train,
              eval_set = [(X_test, y_test)],
              eval_metric="binary_logloss",
              early_stopping_rounds=10,
              verbose=-1,
              callbacks=[LightGBMPruningCallback(trial, metric="binary_logloss")])
        
    preds = model.predict_proba(X_test)
    cv_scores.append(log_loss(y_test, preds))

    return cv_scores

study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X_train_pre, y_train)

n_trials = 100

optim = study.optimize(func, n_trials=n_trials)
print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")


best_params = {'n_estimators': 1000,
               'learning_rate': 0.01,
               'max_depth': 5,
               'boosting_type': 'gbdt'}

lgb_best = lgb.LGBMClassifier(objective="binary",
                          class_weight='balanced', 
                          random_state=23,
                          **best_params)

lgb_best_fit = lgb_best.fit(X_train_pre,
                            y_train,
                            eval_set = [(X_train_pre, y_train), (X_test_pre, y_test)],
                            eval_metric="binary_logloss",
                            verbose=-1,
                            early_stopping_rounds=10)

lgb.plot_metric(lgb_best_fit)

y_pred_test = lgb_best_fit.predict(X_test_pre)
print(classification_report(y_test, y_pred_test))