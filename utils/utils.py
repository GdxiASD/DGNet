# -*- coding:utf-8 -*-
import numpy as np
import torch
import os


def save_model(model, model_dir, epoch=None, val_loss=None):
    """
    Args:
        model: model var.
        model_dir: save path.
        epoch: save model file name.
        val_loss: save model file name.

    Returns:
        return model save path.
    """
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch is not None else 'best'
    if val_loss is not None:
        file_name = os.path.join(model_dir, f'{epoch}_{round(val_loss, 4)}_WXA.pt')
    else:
        file_name = os.path.join(model_dir, f'{epoch}_WXA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)
    return file_name


def MAPE(v, v_, axis=None):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    mape = (np.abs(v_ - v) / (np.abs(v) + 1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    """
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    """
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    """
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    """
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False):
    """
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    """
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
