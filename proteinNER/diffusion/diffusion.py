#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : diffusion.py
# @Author  : zhangchao
# @Date    : 2024/6/26 18:02 
# @Email   : zhangchao5@genomics.cn
import torch
import numpy as np


class SimpleDiffusion:
    def __init__(self, betas):
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, 'Betas must be 1-D'
        assert (betas > 0.).all() and (betas <= 1.).all()

        self.num_timestep = int(betas.shape[0])

        alphas = 1. - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timestep,)

        # calculates for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_one_minus_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculates for posterior q(x_{t-1 | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, timestep, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_start: the initial data batch.
            timestep: the number of diffusion steps (minus 1). Here, 0 means one step.
            noise: if specified, the split-out normal noise.

        Returns:
            A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
                self.extract_into_tensor(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
                + self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape)
                * noise
        )

    def p_sample(self, model, x, timestep):
        """
        Sample x_{t-1} from the model at the given timestep.

        Args:
            model: the model to sample from.
            x: the current tensor at x_{t-1}.
            timestep: the value of timestep, starting at 0 for the first diffusion step.

        Returns:
            a dict containing the following keys:
            - 'sample': a random sample from the model.
            - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, timestep)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (timestep != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out['mean' + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise]
        return {
            'sample': sample,
            'pred_x_start': out['pred_x_start'],
            'greedy_mean': out['mean'],
            'pred_x_start_mean': out['pred_x_start_mean'],
            'out': out
        }

    def p_mean_variance(self, model, x, timestep):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.

        Args:
            model: the model, which takes a signal and a batch of timestep as input.
            x: the [B x L x C] tensor at timestep.
            timestep: a 1-D Tensor of timestep.

        Returns:
            a dict with the following keys:
            - 'mean': the model mean output.
            - 'variance': the model variance output.
            - 'log_variance': the log of 'variance'.
            - 'pred_x_start': the prediction for x_0.
        """

        pred_x_start = model(x, self._scale_timestep(timestep))

        model_variance, model_log_variance = (self.posterior_variance, self.posterior_log_variance_clipped)
        model_variance = self.extract_into_tensor(model_variance, timestep, x.shape)
        model_log_variance = self.extract_into_tensor(model_log_variance, timestep, x.shape)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x_start, x_t=x, timestep=timestep)
        assert (
                model_mean.shape
                == model_log_variance.shape
                == pred_x_start.shape
                == x.shape
        )
        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_x_start': pred_x_start,
            'pred_x_start_mean': pred_x_start
        }

    def q_posterior_mean_variance(self, x_start, x_t, timestep):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        Args:
            x_start:
            x_t:
            timestep:

        Returns:

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self.extract_into_tensor(self.posterior_mean_coef1, timestep, x_t.shape) * x_start
                + self.extract_into_tensor(self.posterior_mean_coef2, timestep, x_t.shape) * x_t
        )
        posterior_variance = self.extract_into_tensor(self.posterior_variance, timestep, x_t.shape)
        posterior_log_variance_clipped = self.extract_into_tensor(
            self.posterior_log_variance_clipped, timestep, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _scale_timestep(self, timestep):
        return timestep.float() * (1000.0 / self.num_timestep)

    def extract_into_tensor(self, array, timestep, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        Args:
            array: the 1-D numpy array.
            timestep: a tensor of indices into the array to extract.
            broadcast_shape: a larger shape of K dimensions with the batch dimension equal to the length of timestep.

        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(array).to(timestep.device)[timestep].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

