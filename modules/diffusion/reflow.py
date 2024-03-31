"""
from https://github.com/yxlllc/DDSP-SVC
MIT License
"""
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from modules.diffusion.wavenet import WaveNet
from modules.diffusion.naive_v2.naive_v2_diff import NaiveV2Diff
from utils.hparams import hparams
REFLOW_DENOISERS = {
    'wavenet': WaveNet,
    'lynxnet': NaiveV2Diff
}


class RectifiedFlow(nn.Module):
    def __init__(self, out_dims, num_feats=1, timesteps=1000, k_step=1000,
                 denoiser_type=None, denoiser_args=None, betas=None,
                 spec_min=None, spec_max=None):
        super().__init__()
        self.velocity_fn: nn.Module = REFLOW_DENOISERS[denoiser_type](out_dims, num_feats, **denoiser_args)
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.loss_type = 'l2_lognorm'
        self.num_feats = num_feats
        self.t_start = (timesteps - k_step) / timesteps
        self.fake_timesteps = timesteps

        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        if self.use_shallow_diffusion:
            assert self.t_start > 0, 'K_step should not be larger than timesteps.'
        self.t_start = self.t_start if self.use_shallow_diffusion else 0.0

    def reflow_loss(self, x_1, t, cond, loss_type=None):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond)

        if loss_type is None:
            loss_type = self.loss_type
        else:
            loss_type = loss_type

        '''
        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)
        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / ( 1 - t)) ** 2)
            loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction='none'))
        else:
            raise NotImplementedError()
        '''
        x_r = x_1 - x_0
        return x_r, v_pred, t

    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, 1000 * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        if infer:
            spec = src_spec
        else:
            spec = gt_spec
        if (self.num_feats == 1) and (gt_spec is not None):
            spec = spec[:, None, :, :]

        x = self.forward_raw(condition, spec, infer, method='euler')

        return x

    def forward_raw(self,
                condition,
                gt_spec=None,
                infer=True,
                infer_step=10,
                method='euler',
                t_start=0.0,
                use_tqdm=True):
        cond = condition.transpose(1, 2)  # [B, H, T]
        b, device = condition.shape[0], condition.device
        depth = hparams.get('K_step_infer', self.k_step)
        if depth is None:
            _t_star = self.t_start
        else:
            _t_star = (self.fake_timesteps - depth)/self.fake_timesteps
        t_start = self.t_start if not infer else max(t_start, _t_star)
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)
            x_1 = x_1.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            t = t_start + (1.0 - t_start) * torch.rand(b, device=device)
            return self.reflow_loss(x_1, t, cond=cond)
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])  # [B, 1, M, T]

            # initial condition and step size of the ODE
            infer_step = self.fake_timesteps // hparams['diff_speedup']
            if gt_spec is None:
                x = torch.randn(shape, device=device)
                t = torch.full((b,), 0, device=device)
                dt = 1.0 / infer_step
            else:
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
                t = torch.full((b,), t_start, device=device)
                dt = (1.0 - t_start) / infer_step

            algorithm = hparams['diff_accelerator']
            if algorithm == 'ddim':
                method = 'euler'
            elif algorithm == 'rk4':
                method = 'rk4'
            elif algorithm == 'euler':
                method = 'euler'
            else:
                raise NotImplementedError(algorithm)
            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)

            elif method == 'rk4':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond)

            else:
                raise NotImplementedError(method)
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]

            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
