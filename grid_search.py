import os
import logging
import time
import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV



def grid_search(data, label, clf, param_grid, cv=None, scoring='f1_macro', n_jobs=-1):
    searcher = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        refit=False,
        verbose=1
    )
    searcher.fit(data, label)
    logging.info("Best parameter (CV score=%0.3f):" % searcher.best_score_)
    logging.info(searcher.best_params_)
    return searcher.best_params_


def initial_param():
    knn_tuned_paramaters = [{"n_neighbors":[3,5,7,10]}]
    logit_tuned_paramaters = [{"solver":['newton-cg', 'lbfgs', 'sag', 'saga'], "C":[0.1, 1, 5, 10]}]
    svm_tuned_parameters = [{'kernel': ['rbf'],'C': [0.1, 1, 5, 10],'gamma': ["auto", 1e-2, 1e-3, 0.1]}]
    dt_tuned_parameters = [{"criterion":["gini","entropy"],"max_features":["auto","log2"],"max_depth":[5,15]}]
    rf_tuned_parameters = [{'n_estimators': [50,100,200,500],"criterion":["gini","entropy"],"max_features":["auto","log2"],"max_depth":[5,15]}]
    xt_tuned_parameters = [{'n_estimators': [50,100,200,500],"criterion":["gini","entropy"],"max_features":["auto","log2"],"max_depth":[5,15]}]
    ada_tuned_parameters =[{"n_estimators":[50,100,200,500],"algorithm":["SAMME", "SAMME.R"],"learning_rate":[1e-3, 1e-2, 0.1, 1]}]
    xgb_tuned_parameters = [{'n_estimators': [50,100,200,500],"max_depth":[5,15],"learning_rate":[1e-3, 1e-2, 0.1, 1],"gamma":[0.01,0.1]}]
    cb_tuned_parameters = [{'iterations':[50,100,200,500], "max_depth":[5,15],"learning_rate":[1e-3, 1e-2, 0.1, 1]}]

    estimators={
        "kNN":KNeighborsClassifier(), 
        "Logit":LogisticRegression(max_iter=400),
        "SVM":SVC(),
        "DecisionTree":DecisionTreeClassifier(),
        "Random_forest": RandomForestClassifier(),
        "ExtraTree": ExtraTreesClassifier(),
        "Adaboost": AdaBoostClassifier(),
        "Xgboost": XGBClassifier(eval_metric='logloss',use_label_encoder =False),
        "Catboost":CatBoostClassifier()
    }
    paramaters={
        "kNN":knn_tuned_paramaters,
        "Logit":logit_tuned_paramaters,
        "SVM":svm_tuned_parameters,
        "DecisionTree":dt_tuned_parameters,
        "Random_forest": rf_tuned_parameters,
        "ExtraTree": xt_tuned_parameters,
        "Adaboost": ada_tuned_parameters,
        "Xgboost": xgb_tuned_parameters,
        "Catboost":cb_tuned_parameters,
    }
    return estimators, paramaters


def run(root_dataset='./dataset',experiment_dir='./experiments/params'):
    root_dir = os.listdir(root_dataset)
    
    for dataset in root_dir: # read dataset name folder
        path_to_dataset = os.path.join(root_dataset, dataset) #dir dataset
        experiment_output = os.path.join(experiment_dir,dataset) #dir experiment
        
        try:
            os.makedirs(experiment_output, exist_ok=True)
        except OSError as e:
            raise

        dataset_files = os.listdir(path_to_dataset)
        for ds in dataset_files:
            if ds.endswith('_C.csv'):
                try:
                    data = pd.read_csv(f"{path_to_dataset}/{ds}", header=0)
                except:
                    data = pd.read_csv(f"{path_to_dataset}/{ds}", header=None)
                print(data)
                X = data[data.columns[:-1]].to_numpy()
                y = data[data.columns[-1]].to_numpy()

                estimators, params = initial_param()
                for estimator_name in estimators.keys():
                    logging.info(estimator_name)
                    best_param = grid_search(
                        data=X,
                        label=y,
                        clf=estimators[estimator_name],
                        param_grid=params[estimator_name],
                        cv=2
                    )
                    with open(f"{experiment_output}/{estimator_name}_params.json", 'w') as f:
                        json.dump(best_param, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    run()
