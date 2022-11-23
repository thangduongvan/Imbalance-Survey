import os
import pandas as pd
import numpy as np
from trainer import Trainer
from utils.load_data import load_kfold
from util_sub.submission_merger import merge_n_submission
from configs.clf_config import config_clfs
from configs.imbl_config import config_imbalance_method


def compute_avarage_folds(dataframe):
    """Compute average final submission and save to csv to rank"""
    lst = []
    for _, df in dataframe.groupby(['imbl','clf']):
        metric_dict = df.mean().to_dict()
        sum_time = df.time_fold.sum()
        std_dict = df.std().to_dict()
        d = {
            'clf': df.clf.tolist()[0],
            'imbl': df.imbl.tolist()[0],
            'dataset': df.dataset.tolist()[0],
            'precision': metric_dict['precision'],
            'std_precision': std_dict['precision'],
            'recall': metric_dict['recall'],
            'std_recall': std_dict['recall'],
            'specificity': metric_dict['specificity'],
            'std_specificity': std_dict['specificity'],
            'fscore': metric_dict['fscore'],
            'std_fscore': std_dict['fscore'],
            'gmean': metric_dict['gmean'],
            'std_gmean': std_dict['gmean'],
            'mcc': metric_dict['mcc'],
            'std_mcc': std_dict['mcc'],
            'balanced_acc': metric_dict['balanced_acc'],
            'std_balanced_acc': std_dict['balanced_acc'],
            'accuracy': metric_dict['accuracy'],
            'std_accuracy': std_dict['accuracy'],
            'time_exec': sum_time
        }
        lst.append(d)
    dframe = pd.DataFrame(lst)#.to_csv(path_to_csv, index=False)
    return dframe


def raw_training(n_training=10, root_dataset='./raw_data', experiment_dir='./experiments/raw', params_clf_dir='./experiments/params', retrain=False):
    lst_path_to_ds = []
    root_dir = os.listdir(root_dataset)
    
    for dataset in root_dir: # read dataset name folder
        print(".................................... ", dataset)
        path_to_dataset = os.path.join(root_dataset, dataset) #dir dataset
        experiment_output = os.path.join(experiment_dir,dataset) #dir experiment
        params_dataset = os.path.join(params_clf_dir,dataset)

        dataset_files = os.listdir(path_to_dataset)
        data_folds = load_kfold(path_to_dataset) # list of fold directory
        
        # training
        trainer = Trainer(
            clfs=config_clfs(),
            imbls=config_imbalance_method(),
            data=data_folds,
            n_training=n_training,
            output_folder=experiment_output,
            params_clf=params_dataset,
            dataset_name=dataset,
            retrain=retrain
        )
        trainer.run()

        # mean of fold in submission.csv
        df_submission = pd.read_csv(experiment_output+'/submission.csv')
        df_mean_fold = compute_avarage_folds(df_submission)
        df_mean_fold.fillna(0, inplace=True)
        df_mean_fold.to_csv(experiment_output+'/mean_submission.csv', index=False)
        del df_submission, df_mean_fold




def sub_training(n_training=1, root_dataset='./sub_data', experiment_dir='./experiments/sub', params_clf_dir='./experiments/params', retrain=False):
    root_dir = os.listdir(root_dataset)
    
    for dataset in root_dir: # read dataset name folder
        print(".................................... ", dataset)
        path_to_dataset = root_dataset+ f'/{dataset}' #dir dataset
        imbalance_dataset_dir = os.listdir(path_to_dataset)
        params_dataset = os.path.join(params_clf_dir,dataset)
        for imbl_ds in imbalance_dataset_dir:
            dataset_name = f'{dataset}_{imbl_ds}'            
            n_dataset_files = os.listdir(path_to_dataset+ f'/{imbl_ds}')
            for ds in n_dataset_files:
                experiment_output = os.path.join(experiment_dir, f'{dataset}/{imbl_ds}/{ds}') #dir experiment
                data_folds = load_kfold(path_to_dataset+f'/{imbl_ds}/{ds}') # list of fold directory
                # tranining
                trainer = Trainer(
                    clfs=config_clfs(),
                    imbls=config_imbalance_method(),
                    data=data_folds,
                    n_training=n_training,
                    output_folder=experiment_output,
                    params_clf=params_dataset,
                    dataset_name = dataset_name,
                    retrain=retrain
                )
                trainer.run()
                del data_folds

            # mean of n-file in sub_data
            cols = ['clf', 'imbl', 'dataset', 'precision', 'recall', 'specificity', 'fscore','gmean', 'mcc', 'balanced_acc', 'accuracy']
            tmp = pd.DataFrame(columns=cols)
            for i, ds in enumerate(n_dataset_files):
                experiment_output = os.path.join(experiment_dir, f'{dataset}/{imbl_ds}/{ds}/submission.csv') #dir experiment
                df_n_submission = pd.read_csv(experiment_output)
                # remove dups and keep last row when there are duplications of clf, imbl, dataset, and fold
                # df_n_submission.drop_duplicates(subset=['clf', 'imbl', 'dataset', 'fold'])
                df_mean_fold = compute_avarage_folds(df_n_submission)
                df_mean_fold.fillna(0, inplace=True)
                tmp = tmp.append(df_mean_fold, ignore_index=True)

            mean_n_df = tmp.groupby(['clf', 'imbl', 'dataset']).mean().reset_index() # mean of n files
            # tmp.sort_values(['clf', 'imbl']).to_csv(os.path.join(experiment_dir, f'{dataset}/{imbl_ds}/tmp_submission.csv'), index=False)
            mean_n_df.to_csv(os.path.join(experiment_dir, f'{dataset}/{imbl_ds}/mean_submission.csv'), index=False) # tinh mean cho nay
            del tmp, df_mean_fold, df_n_submission

    # merge imbalanced ratio submission to unique submission
    merge_n_submission(experiment_dir) #save at project/experiments/sub/breast-cancer/ibml_submission.csv


if __name__ == '__main__':
    # merge_n_submission()
    # raw_training(n_training=10, root_dataset='./raw_data',experiment_dir='./experiments/raw', retrain=False)
    sub_training(n_training=10, root_dataset='./sub_data',experiment_dir='./experiments/sub', retrain=False)