import os
import sys
import time
import json
import pandas as pd
import numpy as np
from utils.logger import setup_logger, clean_console
from utils.save_file import save_pkl
from utils.load_data import cross_validate
from classifier.base import SamplingClassifier
from metrics import get_binary_metrics

import logging
import warnings


warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, clfs, imbls, data, n_training, output_folder, params_clf, dataset_name, retrain=False):
        self.data = data
        self.clfs = clfs
        self.imbls = imbls
        self.n_training = n_training
        self.output_folder = output_folder
        self.params_clf = params_clf
        self.dataset_name = dataset_name
        self.retrain = retrain
        self._create_all_folder()

    def _create_all_folder(self):
        # make folder for experiments
        self._make_folder(self.output_folder)
        
        # make folder for saving pickle file
        self._dir_model = self.output_folder + f'/model'
        self._make_folder(self._dir_model)
        
        # make folder for saving log file
        self._dir_log = self.output_folder + f'/log'
        self._make_folder(self._dir_log)

    def run(self):
        if self.imbls is None:
            return self._train_hybrid_clf()
        else:
            return self._train_imbl_clf()

    def _train_hybrid_clf(self):
        """Only train with classifier"""
        for clf in self.clfs:
            # check params json, and pass param into classifier
            estimator = clf[1]
            path_to_param = f"{self.params_clf}/{clf[0]}_params.json"
            if os.path.isfile(path_to_param):
                with open(f"{self.params_clf}/{clf[0]}_params.json", 'r') as f:
                    params = json.load(f)
                estimator = estimator.set_params(**params)
            
            if self.retrain == False: # check pipline whether the pipeline was trained before
                if os.path.isfile(self._dir_model+f'/{clf[0]}.pkl'):
                    continue
            pipeline = SamplingClassifier(None, clf[1], self.dataset_name)
            # load fold csv and cross validation
            for fold_th, (train, test) in enumerate(cross_validate(self.data)):
                X_train, y_train = train[train.columns[:-1]].to_numpy(), train[train.columns[-1]].to_numpy()
                X_test, y_test = test[test.columns[:-1]].to_numpy(), test[test.columns[-1]].to_numpy()

                # store result for each fold
                res_fold = pd.DataFrame(
                    columns=[
                        'precision',
                        'recall',
                        'fscore',
                        'sensitivity',
                        'specificity',
                        'gmean',
                        'balanced_acc',
                        'mcc',
                        'accuracy'
                    ])
                start_time = time.time()
                # n-loop training
                for n in range(self.n_training):
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    
                    results = get_binary_metrics(y_test, y_pred) # return results
                    res_fold = res_fold.append(
                        {
                            'precision': results['precision'],
                            'recall': results['recall'],
                            'fscore': results['fscore'],
                            'sensitivity': results['sensitivity'],
                            'specificity': results['specificity'],
                            'gmean': results['gmean'],
                            'balanced_acc': results['balanced_acc'],
                            'mcc': results['mcc'],
                            'accuracy': results['accuracy']
                        }
                        ,ignore_index=True
                    )
                # endfor n-loop training
                delta_time = time.time()-start_time
                # result of a fold
                mean_fold = res_fold.mean().to_frame().T
                mean_fold['fold'] = fold_th+1
                mean_fold['clf'] = clf[0]
                mean_fold['imbl'] = None
                mean_fold['dataset'] = self.dataset_name
                mean_fold['time_fold'] = delta_time
                # self.result = self.result.append(mean_fold)
                
                # log process
                setup_logger(logger_name=self.dataset_name, log_file=self._dir_log + f'/{self.dataset_name}.log')
                logger_dataset_name = logging.getLogger(self.dataset_name)
                logger_dataset_name.info("\t{0}\n".format(
                    mean_fold.to_string(
                        header=None,
                        index=False,
                        float_format=lambda x: '{:.4f}'.format(x)
                    ),
                ))
                self._df2csv(df=mean_fold, path=self.output_folder)
            # endfor cross_validate

            logger_dataset_name.info("\t{0}\n".format(pipeline.get_params()))
            save_pkl(pipeline=pipeline, path_to_pkl=self._dir_model+f'/{clf[0]}.pkl') # save model
            clean_console() # clear terminal when too mane lines
        # return self.result

    def _train_imbl_clf(self):
        """Train with imblance method + classifier"""
        for clf in self.clfs:
            # check params json, and pass param into classifier
            estimator = clf[1]
            path_to_param = f"{self.params_clf}/{clf[0]}_params.json"
            if os.path.isfile(path_to_param):
                with open(f"{self.params_clf}/{clf[0]}_params.json", 'r') as f:
                    params = json.load(f)
                estimator = estimator.set_params(**params)

            for imbl in self.imbls:
                if self.retrain == False: # check pipline whether the pipeline was trained before
                    if os.path.isfile(self._dir_model+f'/{imbl[0]}_{clf[0]}.pkl'):
                        continue
                pipeline = SamplingClassifier(imbl[1], estimator, self.dataset_name)
                # load fold csv and cross validation
                for fold_th, (train, test) in enumerate(cross_validate(self.data)):
                    X_train, y_train = train[train.columns[:-1]].to_numpy(), train[train.columns[-1]].to_numpy()
                    X_test, y_test = test[test.columns[:-1]].to_numpy(), test[test.columns[-1]].to_numpy()

                    if np.isnan(X_train.sum()) == True:
                        X_train = np.nan_to_num(X_train)
                    if np.isnan(X_test.sum()) == True:
                        X_test = np.nan_to_num(X_train)

                    # store result for each fold
                    res_fold = pd.DataFrame(
                        columns=[
                            'precision',
                            'recall',
                            'fscore',
                            'sensitivity',
                            'specificity',
                            'gmean',
                            'balanced_acc',
                            'mcc',
                            'accuracy'
                        ])

                    # n-loop training
                    start_time = time.time()
                    for n in range(self.n_training):
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        
                        results = get_binary_metrics(y_test, y_pred) # return results
                        res_fold = res_fold.append(
                            {
                                'precision': results['precision'],
                                'recall': results['recall'],
                                'fscore': results['fscore'],
                                'sensitivity': results['sensitivity'],
                                'specificity': results['specificity'],
                                'gmean': results['gmean'],
                                'balanced_acc': results['balanced_acc'],
                                'mcc': results['mcc'],
                                'accuracy': results['accuracy']
                            }
                            ,ignore_index=True
                        )
                    # endfor n-loop training
                    delta_time = time.time()-start_time
                    # result of a fold
                    mean_fold = res_fold.mean().to_frame().T
                    mean_fold.fillna(0, inplace=True)
                    mean_fold['fold'] = fold_th+1
                    mean_fold['imbl'] = imbl[0]
                    mean_fold['clf'] = clf[0]
                    mean_fold['dataset'] = self.dataset_name
                    mean_fold['time_fold'] = delta_time
                    # self.result = self.result.append(mean_fold)

                    # log process
                    setup_logger(logger_name=self.dataset_name, log_file=self._dir_log + f'/{self.dataset_name}.log')
                    logger_dataset_name = logging.getLogger(self.dataset_name)
                    logger_dataset_name.info("\t{0}\n".format(
                        mean_fold.to_string(
                            header=None,
                            index=False,
                            float_format=lambda x: '{:.4f}'.format(x)
                        ),
                    ))
                    
                    self._df2csv(df=mean_fold, path=self.output_folder)
                # endfor cross_validate
                
                logger_dataset_name.info("\t{0}\n".format(pipeline.get_params()))
                save_pkl(pipeline=pipeline, path_to_pkl=self._dir_model+f'/{imbl[0]}_{clf[0]}.pkl') # save model
            #endfor imbls
            clean_console() # clear terminal when too mane lines
        #endfor clfs
        # return self.result
        
    def _df2csv(self, df, path):
        out_path = os.path.join(path, 'submission.csv')
        if not os.path.isfile(out_path):
            df.to_csv(out_path, index=False)
        else: 
            df.to_csv(out_path, index=False, mode='a', header=False)

    def _make_folder(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise
