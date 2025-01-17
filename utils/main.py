import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from pathlib import Path
import shutil
import random

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

def main(config, logger, exp_dir):

    # Generate noise
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps

    # Create the model
    unet = Guide_UNet(config).cuda()
    traj = np.load('traj_tr_sp.npy')
    head = np.load('head_tr_sp.npy')
    traj = np.swapaxes(traj, 1,2)
    traj = torch.from_numpy(traj).float()
    head = torch.from_numpy(head).float()
    dataset = TensorDataset(traj, head)
    dataloader = DataLoader(dataset,
                            batch_size=config.training.batch_size,
                            shuffle=False)

    # Set up training parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = make_beta_schedule(schedule='cosine', n_timesteps=n_steps, start=args['diffusion']['beta_start'],
                              end=args['diffusion']['beta_end']).cuda()

    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 1e-3
    losses = []
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # Generate new file folder and save model pt
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    xt_list = []
    noise_list = []
    pred_list = []
    for epoch in range(1, config.training.n_epochs + 1):
      for _, (trainx, head) in enumerate(dataloader):
          x0 = trainx.cuda()
          head = head.cuda()
          t = torch.randint(low=0, high=n_steps, size=(len(x0) // 2 + 1, )).cuda()
          t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)].cuda()
          # Get the noised images (xt) and the noise (our target)
          xt, noise = q_xt_x0(x0, t)
          pred_noise = unet(xt.float(), t, head)
          xt_list.append(xt)
          noise_list.append(noise)
          pred_list.append(pred_noise)
          loss = F.mse_loss(noise.float(), pred_noise)
          losses.append(loss.item())
          optim.zero_grad()
          loss.backward()

          optim.step()
          if config.model.ema:
            ema_helper.update(unet)

      if epoch == 100:
          m_path = model_save / f"unet_{epoch}.pt"
          torch.save(unet.state_dict(), m_path)
          m_path = exp_dir / 'results' / f"loss_{epoch}.npy"
          np.save(m_path, np.array(losses))
          # Save the last pred_noise x noise
          pred_noise_np = pred_noise.detach().cpu().numpy()
          noise_np = noise.float().detach().cpu().numpy()
          m_path = exp_dir / 'results' / f"pred_noise_{epoch}.npy"
          np.save(m_path, pred_noise_np)
          m_path = exp_dir / 'results' / f"noise_{epoch}.npy"
          np.save(m_path, noise_np)

    xt_tensor = torch.cat(xt_list, dim=0)
    noise_tensor = torch.cat(noise_list, dim=0)
    pred_tensor = torch.cat(pred_list, dim=0)
    return xt_tensor, noise_tensor, pred_tensor, losses, unet


if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    root_dir = Path(__name__).resolve().parents[0]
    result_name = '{}_steps={}_len={}_{}_bs={}'.format(
        config.data.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)
    exp_dir = root_dir / "DiffTraj" / result_name
    for d in ["results", "models", "logs","Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
      os.makedirs(files_save)

    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)
    xt_, noise_, pred_noise_, loss_, unet = main(config, logger, exp_dir)
