import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def TRAIN(env, model, name, learning_rate=1e-2, loss='mse', pos_only=False, initialise=True,
          learn_friction=False, verbose=2):
    """
    Train a model given an environment and the learning parameters.

    Parameters
    ----------
        env:            Environment
        model:          Model class or model
        learning_rate:  Learning rate
        loss:           Loss type
        initialise:     Initialise the model.
                        If False model will not be reinitialised so that training can continue.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if initialise:
        e = 1.0
        if hasattr(env, 'e'):
            e = env.e
        m = model(env.dt, env.horizon, name=name, dim_state=env.X.shape[1], e=e, pos_only=pos_only,
                  learn_inertia=False, learn_friction=learn_friction, activation='softplus')
    else:
        m = model
    m.to(device)

    X = torch.tensor(env.X.reshape(env.X.shape[0], 1, env.X.shape[1]), dtype=torch.float32)
    c = torch.tensor(env.c.reshape(env.c.shape[0], env.horizon, 1), dtype=torch.float32)

    if pos_only:
        y = torch.tensor(env.y.reshape(env.y.shape[0], env.horizon,
                                       env.X.shape[1])[:, :, :env.X.shape[1]//2], dtype=torch.float32)
    else:
        y = torch.tensor(env.y.reshape(env.y.shape[0], env.horizon, env.X.shape[1]), dtype=torch.float32)
    y = torch.cat([y, c], 2)

    dataset = TensorDataset(X, c, y)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    loss_log = []
    m.train()
    for epoch in range(env.epochs):
        epoch_loss = 0.0
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = m(xb, cb, env.dt, env.horizon)
            loss_val = m.loss_func(yb, preds)
            loss_val.backward()
            optimizer.step()
            epoch_loss += loss_val.item()
        loss_log.append([epoch, epoch_loss / len(loader)])
        if verbose and verbose > 1 and (epoch + 1) % max(1, env.epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{env.epochs}: loss={loss_log[-1][1]:.6f}")

    m.loss_data = np.array(loss_log)
    return m

def PREDICT(env, model):
    """
    Predict the trajectory given the environment and the trained model

    Parameters
    ----------
        env:    Environment
        model:  Trained model
    """
    model.eval()
    with torch.no_grad():
        pred = model.predict_forward(
            torch.tensor([[env.trajectory[0, :-1]]], dtype=torch.float32),
            env.dt,
            env.trajectory.shape[0]
        )[0]
    return pred.cpu()
