import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNetwork

class ResNet(BaseNetwork):
    """ Residual Network """

    def __init__(self, step_size, horizon, name, dim_state, dim_h=500, activation='relu', pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only)

        act = _get_activation(activation)
        self.network = nn.Sequential(
            nn.Linear(dim_state, dim_h),
            act,
            nn.Linear(dim_h, dim_state),
        )

    def step(self, x, c, step_size, t):
        dxdt = self.network(x)
        x_next = x + step_size * dxdt
        c_next = torch.zeros_like(x_next[:, :1])

        return torch.cat([x_next, c_next], 1)


class ResNetContact(ResNet):
    """ Residual Network """

    def __init__(self, step_size, horizon, name, dim_state, dim_h=500, activation='relu', pos_only=True, regularisation=0, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only=pos_only, dim_h=dim_h, activation=activation, **kwargs)

        act = _get_activation(activation)
        self.network = nn.Sequential(
            nn.Linear(dim_state + 1, dim_h),
            act,
            nn.Linear(dim_h, dim_state),
        )

        self.contact = nn.Sequential(
            nn.Linear(dim_state, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def step(self, x, c, step_size, t):
        
        ctf = self.contact(x)
        if c is None:
            c = (ctf.detach() > 0.5).to(ctf.dtype)

        xc = torch.cat([x, c], 1)
        dxdt = self.network(xc)
        x_next = x + step_size * dxdt
        # c_next = tf.zeros_like(x_next[:, :1])

        return torch.cat([x_next, ctf], 1)


    def loss_func(self, y_true, y_pred):
        
        y_true_x = y_true[:, :, :-1]
        y_true_c = y_true[:, :, -1:]
        y_pred_x = y_pred[:, :, :-1]
        y_pred_c = y_pred[:, :, -1:]

        mse = F.mse_loss(y_pred_x, y_true_x, reduction='none').mean(dim=-1).sum(dim=1).mean()
        cent = F.binary_cross_entropy(y_pred_c, y_true_c, reduction='none').mean(dim=-1).sum(dim=1).mean()

        return mse + cent


def _get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        if activation.lower() == 'softplus':
            return nn.Softplus()
        if activation.lower() == 'tanh':
            return nn.Tanh()
    if callable(activation):
        class _Lambda(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)
        return _Lambda(activation)
    return nn.ReLU()
