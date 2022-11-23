from smote_variants import Borderline_SMOTE2, ADASYN, LVQ_SMOTE, ProWSyn, Lee, kmeans_SMOTE, polynom_fit_SMOTE, SMOTE_TomekLinks, NRAS, AMSCO
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE, KMeansSMOTE, RandomOverSampler
import smote_variants

import logging
logging.getLogger(smote_variants.__name__).setLevel(logging.CRITICAL)


def config_imbalance_method():
    imbl_methods = [
        ('pure', None),
        ('EditedNearestNeighbours', EditedNearestNeighbours()),
        ('RandomUnderSampler', RandomUnderSampler()),
        ('RandomOverSampler', RandomOverSampler()),
        ('Tomek', TomekLinks(n_jobs=2)),
        ('NearMiss', NearMiss(n_jobs=2)),
        ('SMOTE', SMOTE(n_jobs=2, k_neighbors=3, random_state=42)),
        ('ADASYN', ADASYN(n_jobs=2, random_state=42)),
        ('Borderline_smote', Borderline_SMOTE2(n_jobs=2, random_state=42)),
        ('Smote_tomek', SMOTE_TomekLinks(n_jobs=2, n_neighbors=3, random_state=42)),
        ('lvq_smote', LVQ_SMOTE(n_jobs=-1, random_state=42)),
        ('ProWSyn', ProWSyn(n_jobs=-1, random_state=42)),
        ('Lee', Lee(n_jobs=-1, random_state=42)),
        ('kmeans_SMOTE', kmeans_SMOTE(n_jobs=-1, random_state=42)),
        ('NRAS', NRAS(random_state=42,  n_neighbors=3)),
        ('AMSCO', AMSCO(random_state=42)),
        ('polynom_fit_SMOTE', polynom_fit_SMOTE(random_state=42)),
    ]

    return imbl_methods

# def config_imbalance_method():
#     cfg_edited_nn = {
#         'sampling_strategy': 'auto',
#         'n_neighbors': 3
#     }

#     cfg_near_miss = {
#         'sampling_strategy': 'auto',
#         'n_neighbors': 3
#     }

#     cfg_tomeklinks = {
#         'sampling_strategy': 'auto'
#     }

#     cfg_smote = {
#         'sampling_strategy': 'auto',
#         'random_state': None,
#         'k_neighbors': 5
#     }

#     cfg_adasyn = {
#         'sampling_strategy': 'auto',
#         'random_state': None,
#         'n_neighbors': 5
#     }

#     cfg_borderline_smote = {
#         'sampling_strategy': 'auto',
#         'random_state': None,
#         'k_neighbors': 5,
#         'n_jobs': -1,
#         'm_neighbors': 10,
#         'kind': 'borderline-1'
#     }

#     cfg_smotetomek = {
#         'sampling_strategy': 'auto',
#         'random_state': None,
#         'smote': None,
#         'tomek': None
#     }

#     imbl_methods = [
#         ('pure', None),
#         ('tomek', TomekLinks(**cfg_tomeklinks)),
#         ('borderline_smote', Borderline_SMOTE2()),
#         ('smote_tomek', SMOTETomek(**cfg_smotetomek))
#     ]

#     return imbl_methods