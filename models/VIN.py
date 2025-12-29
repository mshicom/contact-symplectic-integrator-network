"""
VIN Model
=========

This model is based on the article
"Variational Integrator Networks for Physically Structured Embeddings"
by Steindor Saemundsson, Alexander Terenin, Katja Hofmann, Marc Peter Deisenroth
https://arxiv.org/abs/1910.09349

Code provided by Steindor Saemundsson
Modified by Andreas Hochlehnert to work in this context

Date: August 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNetwork

class VIN(BaseNetwork):
    """ Variational Integrator Network """

    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False, pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only)

        self.dim_Q = self.dim_state // 2

        act = _get_activation(activation)
        self.potential = nn.Sequential(
            nn.Linear(self.dim_Q, dim_h),
            act,
            nn.Linear(dim_h, 1, bias=False),
        )

        self.learn_inertia = learn_inertia
        if self.learn_inertia:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.L_param = nn.Parameter(torch.zeros(num_w))
        else:
            self.L_param = None

    @property
    def M_inv(self):
        if self.learn_inertia:
            L = _fill_triangular(self.L_param, self.dim_Q)
            M_inv = L.t() @ L
        else:
            M_inv = torch.eye(self.dim_Q, dtype=self.potential[0].weight.dtype, device=self.potential[0].weight.device)
        return M_inv

    def grad_potential(self, q):
        with torch.enable_grad():
            q.requires_grad_(True)
            U = self.potential(q).sum()
            return torch.autograd.grad(U, q, create_graph=True)[0]

        
    def loss_func(self, y_true, y_pred):
        y_true_x = y_true[:, :, :-1]
        y_pred_x = y_pred[:, :, :-1]

        mse = F.mse_loss(y_pred_x, y_true_x, reduction='none')
        mse = mse.mean(dim=-1).sum(dim=1).mean()
        return mse

    def step(self, x, c, step_size, t):
        raise NotImplementedError()


class VIN_SV(VIN):
    """ Störmer-Verlet VIN """

    def step(self, x, c, step_size, t):

        c_next = torch.zeros_like(x[:, :1])
        q = x[:, :self.dim_Q]
        q_prev = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)

        qddot = torch.einsum('jk,ik->ij', self.M_inv, dUdq)
        q_next = 2 * q - q_prev - (step_size**2) * qddot

        return torch.cat([q_next, q, c_next], 1)

    def forward(self, q0, step_size, horizon):

        x0 = torch.cat([q0[:, :1], torch.zeros_like(q0[:, :1])], 2)
        x1 = torch.cat([q0[:, 1:2], q0[:, :1]], 2)
        x0 = torch.cat([x0, x1], 1)
        x = [x0]
        for t in range(horizon-1):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, None, step_size, t)
            x.append(x_next[:, None])

        return torch.cat(x, 1)


class VIN_VV(VIN):
    """ Velocity-Verlet VIN """

    def step(self, x, c, step_size, t):

        c_next = torch.zeros_like(x[:, :1])
        q = x[:, :self.dim_Q]
        qdot = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)
        
        qddot = torch.einsum('jk,ik->ij', self.M_inv, dUdq)

        q_next = q + step_size * qdot - 0.5 * (step_size**2) * qddot
        dUdq_next = self.grad_potential(q_next)

        dUdq_mid = dUdq + dUdq_next
        qddot_mid = torch.einsum('jk,ik->ij', self.M_inv, dUdq_mid)

        qdot_next = qdot - 0.5 * step_size * qddot_mid

        return torch.cat([q_next, qdot_next, c_next], 1)


def _fill_triangular(vec, dim):
    L = torch.zeros(dim, dim, device=vec.device, dtype=vec.dtype)
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            L[i, j] = vec[idx]
            idx += 1
    return L


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
