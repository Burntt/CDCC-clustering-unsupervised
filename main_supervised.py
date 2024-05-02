import torch
import importlib
import io
import os
import zipfile
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, rand_score
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
    

def list_file_with_prefix(paths,prefix):
    result=[]
    for data_file in paths:
        s=data_file.split('/')[1]
        if s.startswith(prefix):
            result.append(data_file)
    return result


def parse_data(data_dir):
    if os.path.isfile(data_dir):
        z = zipfile.ZipFile(data_dir, mode='r')
        dir_list = z.namelist()
        path_train = list_file_with_prefix(dir_list, "TRAIN")
        path_test = list_file_with_prefix(dir_list, "TEST")
    else:
        print('data_dir should  be a zip file !')
    train_set = csv_to_X_y(path_train,z)
    test_set = csv_to_X_y(path_test,z)

    # Combine training set data and test set data
    X = np.concatenate((train_set[0], test_set[0]), axis=0)
    y = np.concatenate((train_set[1], test_set[1]), axis=0)
    return (X, y)


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
    set_seed()

    # Hardcoded arguments
    params = {'device': 'cuda', 'epochs': 300, 'lr': 0.01, 'num_layers': 3, 'dropout_rate': 0.5, 'batch_size': 256}
    data = 'CBF'
    full_path = 'data/test/CBF/CBF.zip'

    # Create algorithm object directly from the imported model
    algorithm = CDCCmodel()
    for key, value in params.items():
        setattr(algorithm, key, value)
    print("Created algorithm object with configured parameters.")

    # Set up the model save path, load data, etc.
    model_save = os.path.join('./model_save', data)
    if not os.path.exists(model_save):
        os.makedirs(model_save)
        print("Created model save directory:", model_save)
    model_save_file = os.path.join(model_save, data + '.pt')
    algorithm.model_save_path = model_save_file
    print("Model save path:", model_save_file)

    # Load and process the data
    ds = parse_data(full_path)

    # Set the output dimension of the convolutional layer
    algorithm.CNNoutput_channel = 1152
    print("CNN output channel set to:", algorithm.CNNoutput_channel)

    # Define and evaluate metrics
    metrics = [NMI(), RI()]
    print("Defined metrics:", metrics)

    # Train
    algorithm.train(ds, valid_ds=None, valid_func=metrics)

    # Predict / Trues
    pred = algorithm.predict(ds)
    true_label = np.array(ds[1])

    # Results
    results = [m(true_label, pred) for m in metrics]
    metrics_name = [str(m) for m in metrics]
    results_dict = dict(zip(metrics_name, results))
    print("RESULTS=", results_dict)

if __name__ == '__main__':
    main()