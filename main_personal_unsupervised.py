import torch
import importlib
import io
import os
import zipfile
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, rand_score
from sklearn.metrics import silhouette_score
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import argparse
import warnings
warnings.filterwarnings("ignore")

from algorithm.CDCC.experiment import model as CDCCmodel


class MetricAbstract:
    def __init__(self):
        self.bigger= True
    def __str__(self):
        return self.__class__.__name__
    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")
    

class RI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return rand_score(groundtruth, pred)


class NMI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return normalized_mutual_info_score(groundtruth, pred)
    

def list_file_with_prefix(paths, prefix):
    result = []
    for data_file in paths:
        s = data_file.split('/')[1]
        if s.startswith(prefix):
            result.append(data_file)
    return result

def parse_data_unsupervised(data_dir, prefix="CBF/TRAIN"):
    if os.path.isfile(data_dir):
        with zipfile.ZipFile(data_dir, 'r') as z:
            dir_list = z.namelist()
            feature_files = [f for f in dir_list if f.startswith(prefix) and not f.endswith("_label.csv")]
            arrays = []
            for file in feature_files:
                array = pd.read_csv(z.open(file), header=None).values
                arrays.append(array)

            if arrays:
                X = np.stack(arrays)  # Correctly stacking arrays
                X = np.transpose(X, (0, 2, 1))  # Ensure proper shape for PyTorch: batch, channels, length
                return X
            else:
                raise ValueError("No arrays formed, possibly due to empty file list or read errors.")
    else:
        raise FileNotFoundError(f"Data directory {data_dir} is not a valid zip file!")


def csv_to_X_y(filepath,z):
    list_X = []
    y = None
    for path in filepath:
        dataframe = pd.read_csv(z.open(path), header=None)
        if path.endswith('label.csv'):
            y = np.squeeze(dataframe.values)
        else:
            list_X.append(np.expand_dims(dataframe.values, axis=-1))
    X = np.concatenate(list_X, axis=-1)

    assert (y is not None)
    assert (y.shape[0] == X.shape[0])

    X = np.transpose(X, (0, 2, 1))
    return X, y


def set_seed(seed=2333):
    import random,os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def main():
    set_seed(2333)  # Assuming set_seed function sets the random seed

    params = {'device': 'cuda', 'epochs': 300, 'lr': 0.01, 'num_layers': 3, 'dropout_rate': 0.5, 'batch_size': 256}
    data = 'CBF'
    full_path = 'data/test/CBF/CBF.zip'

    algorithm = CDCCmodel()
    for key, value in params.items():
        setattr(algorithm, key, value)
    print("Algorithm object configured.")

    model_save_dir = './model_save'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, data + '.pt')
    print("Model save path:", model_save_path)

    X = parse_data_unsupervised(full_path)

    print("Starting training...")
    algorithm.train_unsupervised(X)

    embeddings = algorithm.predict(X)
    score = silhouette_score(embeddings, metric='euclidean')
    print("Silhouette Score:", score)

if __name__ == '__main__':
    main()