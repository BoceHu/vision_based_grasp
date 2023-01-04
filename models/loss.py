#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: loss.py

@Date: 2022/12/27
"""
import torch


def bin_focal_loss(pred, target, gamma=2, alpha=0.6):
    n, c, h, w = pred.size()

    _loss = -1 * target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred + 1e-7)
    _gamma = torch.abs(target - pred) ** gamma

    zeros_loc = torch.where(target == 0)
    _alpha = torch.ones_like(pred) * alpha
    _alpha[zeros_loc] = 1 - alpha

    loss = _loss * _gamma * _alpha
    loss = loss.sum() / (n * c * h * w)
    return loss


def focal_loss(net, x, y_pos, y_cos, y_sin, y_wid):
    """
     Focal Loss (FL) is an improved version of Cross-Entropy Loss (CE)
     that tries to handle the class imbalance problem by assigning more weights to hard or easily misclassified examples
     (i.e. background with noisy texture or partial object or the object of our interest )
     and to down-weight easy examples (i.e. Background objects).

     To use focal loss properly, we need ensure each value is in (0,1)
    :param net: GGCNN
    :param x: (batch, 1, h, w)
    :param y_pos: (batch, 1, h, w)
    :param y_cos: (batch, 1, h, w)
    :param y_sin: (batch, 1, h, w)
    :param y_wid: (batch, 1, h, w)
    :return: --
    """
    pred_pos, pred_cos, pred_sin, pred_wid = net(x)

    pred_pos = torch.sigmoid(pred_pos)
    loss_pos = bin_focal_loss(pred_pos, y_pos, alpha=0.9) * 10

    pred_cos = torch.sigmoid(pred_cos)
    loss_cos = bin_focal_loss(pred_cos, (y_cos + 1) / 2, alpha=0.9) * 10

    pred_sin = torch.sigmoid(pred_sin)
    loss_sin = bin_focal_loss(pred_sin, (y_sin + 1) / 2, alpha=0.9) * 10

    pred_wid = torch.sigmoid(pred_wid)
    loss_wid = bin_focal_loss(pred_wid, y_wid, alpha=0.9) * 10

    return {
        'loss': loss_pos + loss_cos + loss_sin + loss_wid,
        'losses': {
            'loss_pos': loss_pos,
            'loss_cos': loss_cos,
            'loss_sin': loss_sin,
            'loss_wid': loss_wid
        },
        'pred': {
            'pred_pos': pred_pos,
            'pred_cos': pred_cos,
            'pred_sin': pred_sin,
            'pred_wid': pred_wid
        }
    }


def get_pred(net, xc):
    net.eval()
    with torch.no_grad():
        pred_pos, pred_cos, pred_sin, pred_wid = net(xc)

        pred_pos = torch.sigmoid(pred_pos)
        pred_cos = torch.sigmoid(pred_cos)
        pred_sin = torch.sigmoid(pred_sin)
        pred_wid = torch.sigmoid(pred_wid)

    return pred_pos, pred_cos, pred_sin, pred_wid
