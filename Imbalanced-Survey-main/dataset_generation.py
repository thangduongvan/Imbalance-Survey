import pandas as pd
import numpy as np
import os
from utils.reader import *


def generate_data(path_csv, lst_ratio=[0.8, 0.6, 0.4, 0.2], folder_save='./sub_data', k=10):
    """Generate imbalanced data acording to imbalance ratio

    Args:
        path_csv (str): path to file csv
        lst_ratio (list, optional): desirable imbalance ratio. Defaults to [0.8, 0.6, 0.4, 0.2].
        folder_save (str, optional): to save file. Defaults to './sub_data'.
        k (int, optional): k-loop sampling to reduce bias. Defaults to 10.

    Usages:
        >> lst_dataset = read_all_datasource_folder('./dataset/')
        >> lst_C_file = read_datasource_C(lst_dataset)
        lst_ratio=[0.9, 0.7, 0.5, 0.3, 0.1]
        >> for f in lst_C_file:
        >>    print("-"*50)
        >>    generate_data(f,lst_ratio, './sub_data', 100)
        >>    print("\n[INFO] Processed {0}\n".format(f))
        >>    print(">"*50)    
    """
    data = pd.read_csv(path_csv)
    
    split_dir = path_csv.split("/")
    fname = split_dir[-1].split('.')[0] #get name dataset 
    parent_dir = folder_save + '/' + fname[:-2] # get parent directory

    n_inst = data[data.columns[-1]].value_counts().to_dict()
    tar_get_col = data.columns[-1]
    mino_samples = min(n_inst.values())
    majo_samples = max(n_inst.values())
    actual_threshold_ir = mino_samples/majo_samples
    minority_class = min(n_inst, key=n_inst.get)
    majority_class = max(n_inst, key=n_inst.get)
    
    print('Ratio: ',lst_ratio)
    print(n_inst)

    for ratio in lst_ratio:
        print("\nRatio: ", ratio)
        parent_dir_ratio = parent_dir + "/" + str(int(ratio*100)) + "/"

        try:
            os.makedirs(parent_dir_ratio, exist_ok=True)
        except OSError as e:
            raise
        
        majo_data = data[data[tar_get_col]==majority_class]
        mino_data = data[data[tar_get_col]==minority_class]
        tmp = []
        # compare actual ratio with desirable ratio to compute n_samples
        if actual_threshold_ir <= ratio:
            n_samples = np.rint(mino_samples/ratio)
            under_data = majo_data.copy()
            keep_data = mino_data.values
        else:
            n_samples = np.rint(majo_samples*ratio)
            under_data = mino_data.copy()
            keep_data = majo_data.values

        # for loop k-times to prevent bias in sampling
        for i in range(k):
            under_data_sampling = under_data.sample(n=int(n_samples)).values
            data_concat0 = np.concatenate([under_data_sampling,keep_data], axis=0)
        
            # create folder for ylabel
            try:
                os.makedirs(parent_dir_ratio + 'n{ith}'.format(ratio=str(int(ratio*100)), ith=i), exist_ok=True)
            except OSError as e:
                raise
            new_fname = "{fname}_{ratio}".format(fname=fname, ratio=str(int(ratio*100)))
            
            #to csv
            path_csv = parent_dir_ratio + 'n{0}/'.format(i) + new_fname + ".csv"
            df = pd.DataFrame(data_concat0)
            
            print(df[df.columns[-1]].value_counts())
            print(path_csv)
            df.to_csv(path_csv, index=False, header=False)
        print('-'*100)


if __name__=="__main__":
    lst_dataset = read_all_datasource_folder('./dataset/')
    lst_C_file = read_datasource_C(lst_dataset)

    lst_ratio=[0.9, 0.7, 0.5, 0.3, 0.1]
    for f in lst_C_file:
        print("-"*50)
        generate_data(f,lst_ratio, './sub_data', 10)
        print("\n[INFO] Processed {0}\n".format(f))
        print(">"*50)