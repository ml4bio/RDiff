import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from random import random
from utils import *
import pdb


def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def gradient_clipping(params, gradnorm_queue, max_grad, disable_log):
    if max_grad <= 1.0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad)
        return

    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
    max_grad_norm = min(max_grad_norm, max_grad)

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if not disable_log:
        if float(grad_norm) > 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std():
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
    return grad_norm


class Queue():
    # from EDM
    # Gradient clipping
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def optimization_manager(config):
    """Return an optimize_fn based on `config`."""

    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    disable_log = config.optim.disable_grad_log

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimize with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            # torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            gradient_clipping(params, gradnorm_queue, grad_clip, disable_log)
        optimizer.step()

    return optimize_fn


def get_step_fn(noise_scheduler, train, optimize_fn, scaler, inverse_scaler, config):
    loss_fn = get_sde_seq_loss_fn(noise_scheduler, train, scaler, inverse_scaler, config)

    def step_fn(state, batch):
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())
        return loss

    return step_fn


def get_sde_seq_loss_fn(noise_scheduler, train, scaler, inverse_scaler, config):
    """The loss function for sequence features."""
    device = config.device
    pred_data = config.model.pred_data
    reduce_mean = config.train.reduce_mean
    self_cond = config.model.self_cond
    batch_size = config.train.batch_size

    def loss_fn(model, batch):
        ##### Note: the batch size has already supported 1 and below #####
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        batch['node_s'] = node_s = scaler(batch['node_s'])

        # sample the noisy samples
        t_eps = 1e-5
        t = torch.rand(batch_size, device=node_s.device) * (1. - t_eps) + t_eps
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)

        noise = torch.randn(node_s.shape, device=node_s.device)
        z_t = expand_dims(alpha_t, node_s.dim()) * node_s + expand_dims(sigma_t, noise.dim()) * noise

        batch['z_t'] = z_t

        noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

        if self_cond:
            assert pred_data
            cond_x = None
            if random() < 0.5:
                with torch.no_grad():
                    cond_x = model(batch, time=t, noise_level=noise_level, cond_x=cond_x)
                    cond_x = cond_x.detach().clone()
            pred = model(batch, time=t, noise_level=noise_level, cond_x=cond_x)
        else:
            pred = model(batch, time=t, noise_level=noise_level)

        length = pred.shape[1]
        if pred_data:
            # data prediction loss
            # losses = torch.square(pred - node_s.squeeze(-2)).mean(dim=-1)
            if pred.shape != node_s.shape:
                losses = torch.square(pred - node_s.squeeze(-2))
            else:
                losses = torch.square(pred - node_s)
            losses = losses * batch['mask'].unsqueeze(-1)
            losses = losses.reshape(batch_size, -1)
            losses = torch.sum(losses, dim=-1)
        else:
            # noise prediction loss (without checking)
            if pred.shape != noise.shape:
                losses = torch.square(pred - noise.squeeze(-2))
            else:
                losses = torch.square(pred - noise)
            losses = losses * batch['mask'].unsqueeze(-1)
            losses = torch.mean(losses, dim=-1)
            losses = losses.reshape(batch_size, -1)
            losses = losses * batch['mask']
            losses = torch.sum(losses, dim=-1)

        if reduce_mean:
            losses = losses / (torch.sum(batch['mask'], dim=-1) + 1e-10)
        else:
            losses = 0.5 * losses

        if pred_data:
            norm = torch.sqrt(alpha_t / sigma_t)
            # norm = (alpha_t / sigma_t) ** 2
            losses = expand_dims(norm, losses.dim()) * losses

        return losses.mean()

    return loss_fn


#################### Utils Functions ####################

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
