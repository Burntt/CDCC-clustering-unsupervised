import torch
from torch.fft import fft
from torch.utils.data import Dataset
import numpy as np
from algorithm.CDCC.augmentations import DataTransform_T, DataTransform_F

class Load_Dataset_Unsupervised(Dataset):
    def __init__(self, model_params, ds):
        super(Load_Dataset_Unsupervised, self).__init__()
        X_train = ds

        # Normalize data
        mean = np.nanmean(X_train)
        std = np.nanstd(X_train)
        X_train = (X_train - mean) / std

        # Convert to Torch tensors
        x_data = torch.from_numpy(X_train).float()

        # Ensure channel dimension is second
        if x_data.shape.index(min(x_data.shape)) != 1:
            x_data = x_data.permute(0, 2, 1)
        
        self.x_data = x_data
        self.len = x_data.shape[0]
        self.x_data_f = fft(x_data).abs()
        
        # Apply augmentations
        self.aug1, self.aug2 = DataTransform_T(x_data, model_params)
        self.aug1_f, self.aug2_f = DataTransform_F(self.x_data_f, model_params)

    def __getitem__(self, index):
        return (self.x_data[index], self.aug1[index], self.aug2[index], 
                self.x_data_f[index], self.aug1_f[index], self.aug2_f[index])

    def __len__(self):
        return self.len

class MyUnsupervisedDataset(Dataset):
    def __init__(self,ds):
        super(MyUnsupervisedDataset, self).__init__()
        X_train = ds[0]
        y_train = ds[1]
        self.len = X_train.shape[0]
        mean = np.nanmean(X_train)
        std = np.nanstd(X_train)
        X_train = (X_train - mean) / std
        if isinstance(X_train, np.ndarray):
            x_data = torch.from_numpy(X_train)
            y_data = torch.from_numpy(y_train).long()
        elif isinstance(X_train, tuple):
            x_data = torch.from_numpy(np.array(X_train))
            y_data = torch.from_numpy(np.array(y_train)).long()
        else:
            x_data = X_train
            y_data = y_train
        self.x_data = x_data
        self.y_data = y_data
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
