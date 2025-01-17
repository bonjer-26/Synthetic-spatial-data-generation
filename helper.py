import torch
from sklearn.datasets import make_swiss_roll

def sample_batch(size, noise=1.0):
    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def make_beta_schedule(schedule='cosine', n_timesteps=500, start=1e-5, end=1e-2):
    if schedule == "cosine":
        t = torch.linspace(0, n_timesteps, n_timesteps + 1) / n_timesteps
        alphas = (torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
    elif schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    return betas
