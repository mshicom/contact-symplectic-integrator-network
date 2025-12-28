import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):

    def __init__(self, step_size, horizon, name, dim_state, pos_only):
        super().__init__()
        self.step_size = float(step_size)
        self.horizon = int(horizon)
        self.dim_state = dim_state
        self.pos_only = pos_only
        self.name = name

    def rollout(self, x0, c_true, step_size=None, horizon=None):
        step = torch.as_tensor(step_size if step_size is not None else self.step_size,
                               device=x0.device, dtype=x0.dtype)
        horizon = int(horizon if horizon is not None else self.horizon)

        x = [x0]
        c = [torch.zeros_like(c_true[:, :1])]
        for t in range(horizon):
            x_t = x[-1][:, -1]
            c_t = c_true[:, t]
            xc_next = self.step(x_t, c_t, step, t)
            x_next = xc_next[:, :-1]
            c_pred = xc_next[:, -1:]
            x.append(x_next[:, None])
            c.append(c_pred[:, None])

        return torch.cat([torch.cat(x, 1), torch.cat(c, 1)], 2)

    def forward(self, x0, c_true, step_size=None, horizon=None):
        xc_out = self.rollout(x0, c_true, step_size, horizon)
        if self.pos_only:
            xpos = xc_out[:, 1:, : self.dim_state // 2]
            return torch.cat([xpos, xc_out[:, 1:, -1:]], 2)
        return xc_out[:, 1:]

    def predict_forward(self, x0, step_size, horizon):
        step = torch.as_tensor(step_size, device=x0.device, dtype=x0.dtype)
        x = [x0]
        c = []
        for t in range(horizon):
            x_t = x[-1][:, -1]
            xc_next = self.step(x_t, None, step, t)
            x_next = xc_next[:, :-1]
            c_pred = xc_next[:, -1:]
            x.append(x_next[:, None])
            c.append(c_pred[:, None])

        c = torch.cat(c, 1)
        c = torch.cat([torch.zeros_like(c[:, :1]), c], 1)
        return torch.cat([torch.cat(x, 1), c], 2)

    def loss_func(self, y_true, y_pred):
        y_true_x = y_true[:, :, :-1]
        y_pred_x = y_pred[:, :, :-1]
        mse = F.mse_loss(y_pred_x, y_true_x, reduction='none')
        mse = mse.mean(dim=-1).sum(dim=1).mean()
        return mse
