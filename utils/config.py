args = {
    'data': {
        'dataset': 'TrajAir',
        'traj_path1': 'traj_tr_sp.npy',
        'head_path2': 'head_tr_kr23.npy',
        'traj_length': 200,
        'channels': 2,
        'uniform_dequantization': False,
        'gaussian_dequantization': False,
        'num_workers': True,
    },
    'model': {
        'type': "simple",
        'attr_dim': 23,
        'guidance_scale': 0.01,
        'in_channels': 2,
        'out_ch': 2,
        'ch': 128,

        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.1,
        'var_type': 'fixedlarge',
        'ema_rate': 0.9999,
        'ema': True,
        'resamp_with_conv': True,
    },
    'diffusion': {
        'beta_schedule': 'cosine',
        'beta_start': 0.0001,
        'beta_end': 0.05,
        'num_diffusion_timesteps': 500,
    },
    'training': {
        'batch_size': 256,
        'n_epochs': 100,
        'n_iters': 100,
        'snapshot_freq': 500,
        'validation_freq': 200,
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    }
}
