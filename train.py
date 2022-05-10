from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import optuna


class Model:
    def __init__(self,model_type,x_filtr,y_train,n_trials):
        self.x_filtr    = x_filtr
        self.y_train    = y_train
        self.n_trials   = n_trials
        self.model_type = model_type

    def cross_score(self,model):
        return cross_val_score(model,self.x_filtr[self.x_filtr.columns[1:]],self.y_train,cv=5,scoring='neg_mean_absolute_error').mean()

    def svm_objective(self,trial):

        C       = trial.suggest_float('C', 1e-3, 10**3, log=True)
        gamma   = trial.suggest_float('gamma',1e-3,1e-1,log=True)
        degree  = trial.suggest_int('degree',2,10)
        model   = SVR(C=C,gamma=gamma,degree=degree)

        return self.cross_score(model)


    def rf_objective(self,trial):

        max_depth = trial.suggest_int('max_depth', 2, 64, log=True)
        #n_estimators = trial.suggest_int("n_estimators", 10, 500)
        #criterion = trial.suggest_categorical('criterion', ['mse', 'mae'])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 4, 20)
        #bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
        max_features = trial.suggest_int('max_features',5,30)

        model = RandomForestRegressor(bootstrap = True, criterion = 'mse',
                                      max_depth = max_depth, max_features = max_features,
                                      max_leaf_nodes = max_leaf_nodes,n_estimators = 200,n_jobs=2)

        return self.cross_score(model)


    def find_params(self,objective):

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params

    def build_model(self):

        if self.model_type == 'SVM':
            model  = SVR()
            params = self.find_params(self.svm_objective)
        else:
            model  = RandomForestRegressor()
            params = self.find_params(self.rf_objective)

        model.set_params(**params)
        model.fit(self.x_filtr[self.x_filtr.columns[1:]],self.y_train)

        return model
