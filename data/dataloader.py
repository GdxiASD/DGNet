from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, raw_data):
        data = raw_data.transpose(1, 2)
        data = data * torch.tensor(self.std).to(data.device) + torch.tensor(self.mean).to(data.device)
        return data.transpose(1, 2)


class min_max_transform:
    def __init__(self, min_value, max_value):
        self.max = max_value
        self.min = min_value

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, raw_data):
        data = raw_data.transpose(1, 2)
        data = data * (self.max - self.min) + self.min
        return data.transpose(1, 2)


# dataset
class Dataset_dim1(Dataset):  # CSV
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test_model', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.type = type
        data = pd.read_csv(root_path).values
        _, Node = data.shape
        if type == '1':
            training_end = int(len(data) * self.train_ratio)
            mms = MinMaxScaler(feature_range=(0,1))
            mms.fit(data[:training_end])
            data_norm = mms.transform(data)
            data = [data_norm]
            self.scaler = mms
        elif type == '2':  # 添加时间信息（5min）
            steps_per_day = 288
            training_end = int(len(data) * self.train_ratio)
            mms = MinMaxScaler(feature_range=(0,1))
            mms.fit(data[:training_end])
            data_norm = mms.transform(data)
            self.scaler = mms
            data = [data_norm.reshape((data_norm.shape[0], Node, 1))]
            tod = [i % steps_per_day /
                   steps_per_day for i in range(data_norm.shape[0])]
            tod = np.array(tod)
            tod_tiled = np.tile(tod, [1, Node, 1]).transpose((2, 1, 0))
            data.append(tod_tiled)
        elif type == '3':  # 添加时间信息（60min）
            steps_per_day = 24
            training_end = int(len(data) * self.train_ratio)
            mms = MinMaxScaler(feature_range=(0,1))
            mms.fit(data[:training_end])
            data_norm = mms.transform(data)
            self.scaler = mms
            data = [data_norm.reshape((data_norm.shape[0], Node, 1))]
            tod = [i % steps_per_day /
                   steps_per_day for i in range(data_norm.shape[0])]
            tod = np.array(tod)
            tod_tiled = np.tile(tod, [1, Node, 1]).transpose((2, 1, 0))
            data.append(tod_tiled)
        else:
            raise 'Dataset Type ERROR!'
        data = np.stack(data, -1).squeeze()
        if self.flag == 'train':
            begin = 0
            end = int(len(data) * self.train_ratio)
            self.trainData = data[begin:end]
            print('train data size:', self.trainData.shape)
        if self.flag == 'val':
            begin = int(len(data) * self.train_ratio)
            end = int(len(data) * (self.val_ratio + self.train_ratio))
            self.valData = data[begin:end]
            print('val data size:', self.valData.shape)
        if self.flag == 'test_model':
            begin = int(len(data) * (self.val_ratio + self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]
            print('test data size:', self.testData.shape)

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData) - self.seq_len - self.pre_len
        elif self.flag == 'val':
            return len(self.valData) - self.seq_len - self.pre_len
        else:
            return len(self.testData) - self.seq_len - self.pre_len

    def inverse_transform(self, data):
        B, N, T = data.shape
        data = data.transpose(1, 2).reshape(-1, N)
        data = self.scaler.inverse_transform(data.cpu())
        data = data.reshape(B, T, N)
        return data.transpose((0, 2, 1))
