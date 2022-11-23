import os
import pandas as pd


def merge_n_submission(ds_root_dir='./experiments/sub'):
    root_dir = os.listdir(ds_root_dir)
    for ds_folder in root_dir:
        path_to_dataset = f'{ds_root_dir}/{ds_folder}'
        imbalance_dataset_dir = os.listdir(path_to_dataset)
        n_submission_lst = []
        for imbl_folder in  imbalance_dataset_dir:
            if imbl_folder != 'imbl_submission.csv':
                n_sub_df = pd.read_csv(f"{path_to_dataset}/{imbl_folder}/mean_submission.csv")
                n_submission_lst.append(n_sub_df)
        imbl_submission = pd.concat(n_submission_lst, ignore_index=True)
        imbl_submission.to_csv(f"{path_to_dataset}/imbl_submission.csv", index=False)
