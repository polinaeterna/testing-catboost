import xgboost as xgb
from hyperopt import hp
from experiment import Experiment
import time


class XGBExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50, 
                 counters_sort_col=None, holdout_size=0, 
                 train_path=None, test_path=None, cd_path=None, num_class=2, output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'xgb', n_estimators, max_hyperopt_evals, 
                            True, counters_sort_col, holdout_size, 
                            train_path, test_path, cd_path, num_class, output_folder_path)

        self.space = {
            'eta': hp.loguniform('eta', -7, 0),
            'max_depth' : hp.quniform('max_depth', 2, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
        }

        self.default_params = {'eta': 0.3, 'max_depth': 6, 'subsample': 1.0, 
                               'colsample_bytree': 1.0, 'colsample_bylevel': 1.0,
                               'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'gamma': 0}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'XGBoost'


    def preprocess_params(self, params):
        if self.learning_task == "classification":
            params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif self.learning_task == "multiclassification":
            params.update({'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'silent': 1, 'num_class': self.num_class})
        elif self.learning_task == "regression":
            params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        params['max_depth'] = int(params['max_depth'])
        return params


    def convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data.astype(float), label)


    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({"seed": seed})
        evals_result = {}
        start_time = time.time()
        bst = xgb.train(params, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)
        eval_time = time.time() - start_time
        
        if self.learning_task == 'regression':
            results = evals_result['test']['rmse']
        elif self.learning_task == 'classification':
            results = evals_result['test']['logloss']
        elif self.learning_task == 'multiclassification':
            results = evals_result['test']['mlogloss']
            
        return bst, results, eval_time


    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest)
        return preds
