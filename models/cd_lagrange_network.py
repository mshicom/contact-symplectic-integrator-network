"""
CD-Lagrange Network
===================

This is the implementation of the CD-Lagrange network based on the work of
- Jean  Di  Stasio  et  al.
  “Benchmark  cases  for  robust  explicit  time  integrators  in  non-smooth  transient  dynamics”
- Fatima-Ezzahra Fekak et al.
  “A new heterogeneous asynchronous explicit–implicit timeintegrator for nonsmooth dynamics”
- Steindor Saemundsson et al.
  “Variational Integrator Networks for Physically Meaning-ful Embeddings”
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNetwork

class CDLNetwork(BaseNetwork):
    """
    CD-Lagrange Network
    ===================

    step_size:      `float`;
                    The time step size "h" used for integration.
    horizon:        `int`;
                    The number of forward steps the network has to predict during training.
    name:           `str`;
                    Name of the network
    dim_state:      `int`;
                    Number of states the system has.
    dim_h:          `int`;
                    Number of units in the hidden layer.
    activation:     `str`;
                    Activation function used in the potential network.
    learn_inertia:  `bool`;
                    Determines whether to learn the mass matrix
    learn_friction: `bool`;
                    Determines whether to learn the friction paramter.
    """

  
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='tanh',
                learn_inertia=False, learn_friction=False, pos_only=True, regularisation=5e-6, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only)

        self.dim_Q = self.dim_state // 2
        
        act = _get_activation(activation)
        self.potential = nn.Sequential(
            nn.Linear(dim_state // 2, dim_h),
            act,
            nn.Linear(dim_h, 1, bias=False),
        )

        self.contact = nn.Sequential(
            nn.Linear(dim_state, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )
        
        self.learn_friction = learn_friction
        if self.learn_friction:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.B_param = nn.Parameter(torch.zeros(num_w))
        else:
            self.B_param = None

        self.learn_inertia = learn_inertia
        if self.learn_inertia:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.M_param = nn.Parameter(torch.zeros(num_w))
        else:
            self.M_param = None

        # self.L = tf.constant([[1., -1.]])
        self.register_buffer("L", torch.tensor([[1.]], dtype=torch.float32))
        self.register_buffer("e", torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float32))


    @property
    def B(self):
        """Friction paramter"""
        if self.learn_friction:
            B = _fill_triangular(self.B_param, self.dim_Q)
        else:
            B = torch.zeros(self.dim_Q, self.dim_Q, device=self.L.device, dtype=self.L.dtype)
        return B
    
    @property
    def M_inv(self):
        """Inverse of mass matrix"""
        if self.learn_inertia:
            L = _fill_triangular(self.M_param, self.dim_Q)
            M_inv = L.t() @ L
        else:
            M_inv = torch.eye(self.dim_Q, device=self.L.device, dtype=self.L.dtype)
        return M_inv

    def grad_potential(self, q):
        """Gradient of the potential"""
        q.requires_grad_(True)
        U = self.potential(q).sum()
        return torch.autograd.grad(U, q, create_graph=True)[0]

    def step(self, x, c, step_size, t):
        """Calculate next step using the CD-Lagrange integrator.""" 
        u = x[:, :self.dim_Q] 
        udot = x[:, self.dim_Q:] 

        u_next = u + step_size * udot 
        dUdu = self.grad_potential(u_next) 
        damping = torch.einsum('jk,ik->ij', self.B, udot) 
        w = torch.einsum('jk,ik->ij', self.M_inv,  step_size * (dUdu - damping)) 
        
        
        # Contact forces 
        Q = torch.cat([u_next, udot], 1)

        v = torch.einsum('jk,ik->ij', self.e, self.L * udot) 
        r = v - self.L * (udot + w) 
        ctf = self.contact(Q) 

        if c is None: 
            c = (ctf.detach() > 0.5).to(ctf.dtype)


        #closest point projection
        u_next = torch.where(ctf > 0.5, torch.zeros_like(u_next), u_next)
        r = c * r

        i = torch.einsum('jk,ik->ij', self.M_inv, self.L * r) 

        # Velocity next step 
        udot_next = udot + w + i
        
        return torch.cat([u_next, udot_next, ctf], 1)

    def loss_func(self, y_true, y_pred):

        y_true_x = y_true[:, :, :-1]
        y_true_c = y_true[:, :, -1:]
        y_pred_x = y_pred[:, :, :-1]
        y_pred_c = y_pred[:, :, -1:]

        mse = F.mse_loss(y_pred_x, y_true_x, reduction='none').mean(dim=-1).sum(dim=1).mean()
        cent = F.binary_cross_entropy(y_pred_c, y_true_c, reduction='none').mean(dim=-1).sum(dim=1).mean()

        return cent + mse


class CDLNetwork_Simple(CDLNetwork):
    
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False,
                 learn_friction=False, e=1., pos_only=True, regularisation=5e-3, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, dim_h, activation, learn_inertia, learn_friction,
                         pos_only, regularisation)

        # self.contact = tfk.Sequential([
        #     tfk.layers.Dense(dim_h, activation='relu'),
        #     tfk.layers.Dense(dim_state//2, activation='sigmoid')
        # ])
        
        self.register_buffer("L", torch.tensor([[1.]], dtype=torch.float32))
        self.register_buffer("e", e * torch.eye(self.dim_Q, dtype=torch.float32))

    def step(self, x, c, step_size, t): 
        """Calculate next step using the CD-Lagrange integrator.""" 
        u = x[:, :self.dim_Q] 
        udot = x[:, self.dim_Q:] 

        u_next = u + step_size * udot 
        dUdu = self.grad_potential(u_next) 
        damping = torch.einsum('jk,ik->ij', self.B, udot) 
        w = torch.einsum('jk,ik->ij', self.M_inv,  step_size * (dUdu - damping)) 
        
        
        # Contact forces
        Q = torch.cat([u_next, udot], 1)

        v = -self.e * self.L * udot
        r = v - self.L * (udot + w) 
        ctf = self.contact(Q) 

        if c is None: 
            c = (ctf.detach() > 0.5).to(ctf.dtype)

        r = c * r  

        i = torch.einsum('jk,ik->ij', self.M_inv, self.L * r) 

        # Velocity next step 
        udot_next = udot + w + i 
        
        return torch.cat([u_next, udot_next, ctf], 1)

class CDLNetwork_NoContact(CDLNetwork_Simple):
    
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False,
                 learn_friction=False, e=1., **kwargs):
        super().__init__(step_size, horizon, name, dim_state, dim_h, activation, learn_inertia,
                         learn_friction, e)
        
        self.contact = lambda a: torch.zeros_like(a)

        
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
