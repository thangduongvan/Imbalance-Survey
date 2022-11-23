# How to work

## Environment for linux
- Python ver 3.8.10

- Create a virtual-environment ```python -m venv .env```
- ```pip install -r requirements.txt```
- Activate environment ```source .env/bin/activate```

## Experiment

- Set ```ulimit -Sn 1000000``` for opening many files. (For Linux system)
- In training.py:
    - Run: ```sub_training(n_training=1, root_dataset='./sub_data',experiment_dir='./experiments/sub', retrain=False)``` for sub_data folder
    - Run   ```raw_training(n_training=1, root_dataset='./raw_data',experiment_dir='./experiments/raw', retrain=False)``` for raw_data folder
    - ```n_training```: n-loop for training
    
    ```python training.py``` run experiments
- Add classifiers in configs/clf_config.py
- Add imbalanced methods in configs/imbl_config.py


## Data splitting

To split fold data, run ```split_data.py``` file.

For raw:
    ```python --path_to_data_folder './raw_data' --kfold 5```

For sub:
    ```python --raw False --path_to_data_folder './sub_data' --kfold 5```
