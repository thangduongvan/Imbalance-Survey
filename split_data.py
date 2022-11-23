import csv
from utils.reader import read_and_handle, read_sub_datasource, read_raw_datasource
import argparse
import pandas as pd



def split_data(data: pd.DataFrame(), n_splits:int):
    result_df = pd.DataFrame()
    label = data.columns[-1]
    min_value = min(data[label].value_counts())
    # print(data[label].value_counts())
    if n_splits > min_value:
        raise Exception('Cannot have number of splits n_splits=' + str(n_splits) +
                        ' greater than the number of samples: n_samples=' + str(min_value))
    for i in data[label].unique():
        tmp=data[data[label] == i]
        idx = int(round(tmp.shape[0]/n_splits,0))
        tmp = tmp.sample(frac = 1)
        tmp = tmp.reset_index()
        tmp = tmp.loc[:idx-1,:].drop("index",axis=1)
        result_df = pd.concat([result_df,tmp])
    return result_df.sample(frac = 1).reset_index().drop('index',axis=1)


def split_sub_data(folder_path='./sub_data', kfold=5):
    all_sub_csv = read_sub_datasource(folder_path)
    for path_each_file in all_sub_csv:
        split_path = path_each_file.split('/')
        dataset_name = split_path[-1] # breast-cancer_C_10.csv
        path_to_folder = '/'.join(split_path[:-1]) #'./sub_data/breast-cancer/10/n0'
        if not dataset_name.startswith('fold'):
            df = read_and_handle(path=path_each_file)
            for i in range(kfold):
                sample_df = split_data(data=df, n_splits=kfold)
                sample_df.to_csv(
                    path_to_folder + '/' + 'fold_' + str(i) + '.csv',
                    index=False,
                    header=None
                )


def split_raw_data(folder_path='./raw_data', kfold=5):
    all_sub_csv = read_raw_datasource(folder_path)
    for path_each_file in all_sub_csv:
        split_path = path_each_file.split('/')
        dataset_name = split_path[-1] # breast-cancer_C_10.csv
        path_to_folder = '/'.join(split_path[:-1]) #'./raw_data/breast-cancer/'
        if (not dataset_name.startswith('fold')) and (dataset_name.endswith('_C.csv')):
            df = read_and_handle(path=path_each_file)
            for i in range(kfold):
                sample_df = split_data(data=df, n_splits=kfold)
                sample_df.to_csv(
                    path_to_folder + '/' + 'fold_' + str(i) + '.csv',
                    index=False,
                    header=None
                )


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--raw', default=True, type=bool)
    arg.add_argument('--path_to_data_folder', default='./raw_data', type=str)
    arg.add_argument('--kfold', default=5, type=int)


    args = arg.parse_args()
    is_raw = args.raw
    path_to_data_folder = args.path_to_data_folder
    kfolds = args.kfold

    if is_raw:
        split_raw_data(folder_path=path_to_data_folder, kfold=kfolds)
    else:
        split_sub_data(folder_path=path_to_data_folder, kfold=kfolds)
