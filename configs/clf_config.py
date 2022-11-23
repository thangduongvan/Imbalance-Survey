from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import logging
logging.getLogger(XGBClassifier.__name__).setLevel(logging.CRITICAL)
logging.getLogger(CatBoostClassifier.__name__).setLevel(logging.CRITICAL)


def config_clfs():
    clfs = [
        ('kNN', KNeighborsClassifier()),
        ('Logit', LogisticRegression()),
        ('SVM', SVC()),
        ('DecisionTree', DecisionTreeClassifier()),
        ('Random_forest', RandomForestClassifier()),
        ('ExtraTree', ExtraTreesClassifier()),
        ('Adaboost', AdaBoostClassifier()),
        ('Xgboost', XGBClassifier(verbosity=0, silent=True)),
        ('Catboost', CatBoostClassifier(verbose=0)),
    ]

    return clfs

# def config_clfs():
#     cfg_5nn = {
#         'n_neighbors': 5,
#         'metric': 'euclidean',
#         'n_jobs': -1,
#     }

#     cfg_rf = {
#         'n_estimators': 100,
#         'criterion': 'gini',
#         'max_features': 'auto',
#         'n_jobs': None,
#         'random_state': None,
#         'class_weight': None
#     }

#     clfs = [
#         ('kNN', KNeighborsClassifier(**cfg_5nn)),
#         ('Random_forest', RandomForestClassifier(**cfg_rf)),
#     ]

#     return clfs