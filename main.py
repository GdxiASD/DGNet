import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import Dataset_dim1
from model.model import *
import time
import os
import numpy as np
from utils.utils import save_model, load_model, evaluate, loss
import pandas as pd
from fig import save_func

# main settings can be seen in markdown file (README.md)
parser = argparse.ArgumentParser(description='Adaptive ConvNet for space-time data forecasting')
parser.add_argument('--exid', type=str, default='1', help='experiment id')
# data setting
parser.add_argument('--model', type=str, default='DGNet', help='')
parser.add_argument('--data', type=str, default='metr', help='data set')
parser.add_argument('--train_ratio', type=float, default=0.7, help='')
parser.add_argument('--val_ratio', type=float, default=0.2, help='')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--inverse', type=bool, default=False, help='inverse transform')
# train setting
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input data batch size')
parser.add_argument('--validate_freq', type=int, default=1, help='')
parser.add_argument('--early_stop', type=bool, default=True, help='')
parser.add_argument('--seed', type=int, default=20231121, help='random seed')
# learning rate scheduler
parser.add_argument('--learning_rate', type=float, default=2e-04, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=25, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
# model parameter
parser.add_argument('--feature_size', type=int, default=207, help='feature size')
parser.add_argument('--seq_length', type=int, default=12, help='input length')
parser.add_argument('--pre_length', type=int, default=12, help='predict length')
parser.add_argument('--in_dim', type=int, default=1, help='')
parser.add_argument('--embed_size', type=int, default=32, help='embedding dimensions')
parser.add_argument('--hidden_size', type=int, default=32, help='hidden dimensions')
parser.add_argument('--order', type=int, default=2, help='diffusion steps')
parser.add_argument('--linear', type=int, default=3, help='MLP layer num')
parser.add_argument('--blocks', type=int, default=6, help='model stack num')
parser.add_argument('--scale', type=int, default=1, help='adaptive weight scale')

# model switch
parser.add_argument('--bi', default=False, action='store_true', help='')
parser.add_argument('--GCN', default=False, action='store_true', help='use GCN')
parser.add_argument('--TCN', default=False, action='store_true', help='use TCN')
parser.add_argument('--blocks_gate', default=False, action='store_true', help='use blocks_gate')
parser.add_argument('--graph_regenerate', default=False, action='store_true', help='use GLL')
parser.add_argument('--is_graph_shared', default=False, action='store_true', help='is shared graph')

# adaptive matrix setting
parser.add_argument('--alpha', type=int, default=3, help='tanh activation expansion rate')
parser.add_argument('--dim_time_emb', type=int, default=64, help='')
parser.add_argument('--dim_graph_emb', type=int, default=64, help='')
args = parser.parse_args()
print(f'Training configs: {args}')
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
# create output dir

setting = 'id-{}_fs{}_pl{}_sl{}_ems{}_hs{}_bas{}_K{}_b{}_ab{}_bi{}_indim{}_gr{}_gs{}'.format(
    args.exid,
    args.feature_size,
    args.pre_length,
    args.seq_length,
    args.embed_size,
    args.hidden_size,
    args.batch_size,
    args.order,
    args.blocks,
    f'{args.alpha}',
    args.bi,
    args.in_dim,
    args.graph_regenerate,
    args.is_graph_shared,
)
result_train_file = os.path.join('output', args.data, setting)
print(result_train_file)
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)

# data set
data_parser = {
    'Electricity': {'root_path': 'data/Electricity.csv', 'type': '3'},
    'metr': {'root_path': 'data/metr.csv', 'type': '1'},
    'pems07': {'root_path': 'data/PeMS07.csv', 'type': '2'},
    'pems03': {'root_path': 'data/PeMS03.csv', 'type': '2'},
}

# data process
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
else:
    raise ValueError('Data set does not exist')
data_dict = {
    'Electricity': Dataset_dim1,
    'pems07': Dataset_dim1,
    'pems03': Dataset_dim1,
    'metr': Dataset_dim1,
}

Data = data_dict[args.data]
# train val test_model
train_set = Data(root_path=data_info['root_path'], flag='train', seq_len=args.seq_length, pre_len=args.pre_length,
                 type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)
test_set = Data(root_path=data_info['root_path'], flag='test_model', seq_len=args.seq_length, pre_len=args.pre_length,
                type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)
val_set = Data(root_path=data_info['root_path'], flag='val', seq_len=args.seq_length, pre_len=args.pre_length,
               type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)

train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

model = DGNetv2(pre_length=args.pre_length, embed_size=args.embed_size, feature_size=args.feature_size,
                seq_length=args.seq_length, hidden_size=args.hidden_size, device=device,
                graph_regenerate=args.graph_regenerate, in_dim=args.in_dim, linear=args.linear, scale=args.scale,
                order=args.order, blocks=args.blocks, dim_time_emb=args.dim_time_emb,
                dim_graph_emb=args.dim_graph_emb, GCN=args.GCN, TCN=args.TCN, blocks_gate=args.blocks_gate,
                is_graph_shared=args.is_graph_shared, alpha=args.alpha)
parameter_number = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(" Parameters Volume: ", parameter_number)

my_optim = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, eps=1e-08, weight_decay=1e-4)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
forecast_loss = nn.L1Loss(reduction='mean').to(device)


def validate(model, vali_loader):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        forecast = model(x)
        if len(y.shape) == 3:
            y = y.permute(0, 2, 1).contiguous()
        elif len(y.shape) == 4:
            y = y[..., 0]
            y = y.permute(0, 2, 1).contiguous()
        if args.inverse:
            forecast = train_set.inverse_transform(forecast)
            y = train_set.inverse_transform(y)
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'Validate: Average--all. MAPE {score[0]:7.9%}, MAE {score[1]:7.9f}, RMSE {score[2]:7.9f}.')
    model.train()
    return loss_total / cnt, score


def test(absolute_path=None, best_model_var=None):
    if absolute_path is None and best_model_var is not None:
        model = best_model_var
    else:
        model = torch.load(absolute_path)
    model.eval()
    preds = []
    trues = []
    for index, (x, y) in enumerate(test_dataloader):
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        forecast = model(x)
        if len(y.shape) == 3:
            y = y.permute(0, 2, 1).contiguous()
        elif len(y.shape) == 4:
            y = y[..., 0]
            y = y.permute(0, 2, 1).contiguous()
        if args.inverse:
            forecast = train_set.inverse_transform(forecast)
            y = train_set.inverse_transform(y)
        forecast = forecast.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        preds.append(forecast)
        trues.append(y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds, by_node=True)
    for i in [2, 5, 11]:
        print(f'Test: Horizon--{i + 1}. MAPE {score[0][i]:7.4%}, MAE {score[1][i]:7.4f}, RMSE {score[2][i]:7.4f}.')
    score = evaluate(trues, preds)
    print(f'Test: Average--all. MAPE {score[0]:7.4%}, MAE {score[1]:7.4f}, RMSE {score[2]:7.4f}.')
    return score


if __name__ == '__main__':
    val_min_loss = None
    best_model = None
    best_model_inf = {}
    best_epoch = 0
    val_loss = None
    val_loss_list = []
    train_loss = []
    runtime_list = []

    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            my_optim.zero_grad()
            cnt += 1
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x)
            if len(y.shape) == 3:
                y = y.permute(0, 2, 1).contiguous()
            elif len(y.shape) == 4:
                y = y[..., 0]
                y = y.permute(0, 2, 1).contiguous()
            if args.inverse:
                forecast = train_set.inverse_transform(forecast)
                y = train_set.inverse_transform(y)
            loss = forecast_loss(forecast, y)
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        if (epoch + 1) % args.exponential_decay_step == 0 or epoch - best_epoch > 10:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss, score = validate(model, val_dataloader)
            val_loss_list.append(val_loss)
            if val_min_loss is None:
                val_min_loss = val_loss
                val_min_score = score
                train_min_loss = loss_total / cnt
                best_model = model
                best_epoch = epoch
            elif val_loss < val_min_loss:
                val_min_loss = val_loss
                val_min_score = score
                best_model = model
                train_min_loss = loss_total / cnt
                best_epoch = epoch
                test(best_model_var=best_model)
        runtime_list.append(time.time() - epoch_start_time)
        print('Epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | val_loss {:5.6f} | best epoch {}'.format(
                epoch, runtime_list[-1], loss_total / cnt, val_loss, best_epoch))
        train_loss.append(loss_total / cnt)
        save_model(model, result_train_file+r'./train', epoch=epoch, val_loss=val_loss)
        if epoch - best_epoch > 25 and args.early_stop:
            print('early stop! Epoch: {}'.format(epoch))
            break
    mean_runtime = np.mean(np.array(runtime_list))
    print(f'epoch:{best_epoch}. loss:{val_min_loss}. mean runtime:{mean_runtime}')
    best_model_file_name = save_model(best_model, result_train_file+r'./train', val_loss=val_min_loss)
    score = test(absolute_path=best_model_file_name)
