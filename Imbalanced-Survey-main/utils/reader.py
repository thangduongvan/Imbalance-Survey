import numpy as np
import pandas as pd
import os
import csv


def read_all_datasource_folder(path):
    """
    Args:
        path: path to datasource (parent folder)

    Return:
        a list contains subfolder

    Usage:
        >> read_all_datasource('./dataset')
        ['./dataset/acute-inflammations',
        './dataset/breast-cancer',
        './dataset/breast-cancer-coimbra',
        './dataset/caesarian',
        ...]        
    """
    return [sub[0].replace('\\','/') for sub in os.walk(path)][1:] #first element is parent folder


def read_datasource_C(lst_dataset):
    lst_path_to_C_file = []
    for dataset in lst_dataset:
        lst_file = os.listdir(dataset)
        for file in lst_file:
            if file.endswith("C.csv"):
                lst_path_to_C_file.append(str(dataset)+"/"+str(file))
    return lst_path_to_C_file


def read_raw_datasource(root_dataset):
    """Read raw data

    Args:
        root_dataset (str): path to sub-folder (e.g: './raw_data')

    Returns:
        list: list of raw *csv

    Usages:
    >> read_raw_datasource('./dataset')
    ['./dataset/breast-cancer/breast-cancer_C.csv',
    './dataset/breast-cancer-coimbra/dataR2_C.csv',
    './dataset/caesarian/caesarian_C.csv',
    './dataset/cervical-cancer-behavior/cervical_C.csv',
    './dataset/cervical-cancer-risk/risk_factors_cervical_cancer_C.csv',
    './dataset/diabetes/diabetes_data_upload_C.csv',
    './dataset/diabetic-retinopathy-debrecen/messidor_C.csv',
    './dataset/eeg-eye-state/eeg-eye-state_C.csv',
    './dataset/haberman/haberman_C.csv',
    './dataset/heart_failure_clinical_records_dataset/heart_failure_clinical_records_dataset_C.csv',
    './dataset/parkinson/parkinson_C.csv',
    './dataset/sonar/sonar_C.csv']
    """
    lst_path_to_ds = []
    root_dir = os.listdir(root_dataset)
    
    for dataset in root_dir: # read dataset name folder
        path_to_dataset = root_dataset+ f'/{dataset}'
        dataset_file = os.listdir(path_to_dataset)
        for f in dataset_file: # read dataset_A/dataset_B/dataset_C file
            if f.endswith(f"_C.csv"):
                lst_path_to_ds.append(path_to_dataset+f'/{f}') # store all csv
                break
    return lst_path_to_ds


def read_sub_datasource(root_dataset):
    """Read sub data

    Args:
        root_dataset (str): path to sub-folder (e.g: './sub_data')

    Returns:
        list: list of raw *csv

    Usages:
    >> read_sub_datasource('./dataset')
    ['./sub_data/breast-cancer/10/n0/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n1/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n2/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n3/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n4/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n5/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n6/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n7/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n8/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/10/n9/breast-cancer_C_10.csv',
    './sub_data/breast-cancer/30/n0/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n1/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n2/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n3/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n4/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n5/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n6/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n7/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n8/breast-cancer_C_30.csv',
    './sub_data/breast-cancer/30/n9/breast-cancer_C_30.csv',]
    """
    lst_path_to_ds = []
    root_dir = os.listdir(root_dataset)
    
    for dataset in root_dir: # read dataset name folder
        path_to_dataset = root_dataset+ f'/{dataset}'
        ratio_dataset = os.listdir(path_to_dataset)
        for ratio in ratio_dataset: # read ratio folder
            path_to_ratio = path_to_dataset + f'/{ratio}'
            n_dataset = os.listdir(path_to_ratio)
            for ith in n_dataset: # read nth sampling dataset
                path_to_n_dataset = path_to_ratio + f'/{ith}'
                for csv_f in os.listdir(path_to_n_dataset):
                    if csv_f.endswith(f"_C_{ratio}.csv"):  
                        lst_path_to_ds.append(path_to_n_dataset+f'/{csv_f}') # store all csv
    return lst_path_to_ds


def read_and_handle(path):
    sample = open(path, 'r').read()
    sniffer = csv.Sniffer()
    separator = sniffer.sniff(sample).delimiter
    header = sniffer.has_header(sample)
    if header:
        df_raw = pd.read_csv(path, header=None, sep=separator)
    else:
        df_raw = pd.read_csv(path, header=None, sep=separator)
    return df_raw