from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
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


    def gbm_objective(self,trial):

        max_depth = trial.suggest_int('max_depth', 2, 10)

        max_features = trial.suggest_int('max_features',5,40)

        lr = trial.suggest_float('learning_rate',0.05,0.1,log=True)

        n_est = trial.suggest_int('n_estimators',50,300)

        subs = trial.suggest_float('subsample',0.2,1)

        model = GradientBoostingRegressor(learning_rate=lr,n_estimators=n_est,subsample=subs,max_depth=max_depth,max_features=max_features)

        return self.cross_score(model)


    def mlp_objective(self,trial):

        lr    = trial.suggest_float('learning_rate_init',10**-5,10**-2,log=True)

        alpha = trial.suggest_float('alpha',10**-4,10**-1,log=True)

        bz    = trial.suggest_int('batch_size',3,10)

        model = MLPRegressor(hidden_layer_sizes=(100,100),learning_rate_init=lr,alpha=alpha,batch_size=bz,max_iter=1000,tol=1e-2)

        return self.cross_score(model)


    def elastic_net(self,trial):

        l1_ratio = trial.suggest_float('l1_ratio',0,1)
        alpha    = trial.suggest_float('alpha',10**-2,10**2,log=True)
        model    = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,tol=1e-2)

        return self.cross_score(model)

    def find_params(self,objective):

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params

    def build_model(self):

        if self.model_type == 'SVM':
            model  = SVR()
            params = self.find_params(self.svm_objective)
        elif self.model_type == 'RF':
            model  = RandomForestRegressor()
            params = self.find_params(self.rf_objective)

        elif self.model_type == 'EN':
            params = self.find_params(self.elastic_net)
            model  = ElasticNet()

        elif self.model_type =='GBM':
            params = self.find_params(self.gbm_objective)
            model  = GradientBoostingRegressor()

        else:
            model = MLPRegressor()
            params = self.find_params(self.mlp_objective)

        model.set_params(**params)
        model.fit(self.x_filtr[self.x_filtr.columns[1:]],self.y_train)

        return model
