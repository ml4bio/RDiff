import torch


def post_process(gen_seq, inverse_scaler):
    """Post process generated sequences."""
    gen_seq = inverse_scaler(gen_seq)
    gen_seq = torch.argmax(gen_seq, dim=-1)
    return gen_seq

def get_sampling_fn(config, noise_scheduler, sampling_steps, inverse_scaler, eps=1e-3):
    device = config.device

    time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
    sampler = AncestralSampler(config, noise_scheduler, time_steps, config.model.pred_data, inverse_scaler)
    # n_samples = config.eval.n_samples

    @torch.no_grad()
    def sampling_fn(model, data, fixed_part=False):
        model.eval()
        # extend the sequence into a batch according to n_samples
        batch = data
        # sample initial noise
        z = torch.randn(batch['node_s'].shape, device=device)
        if fixed_part:
            gen_seq = sampler.sampling_fixed(model, z, batch)
        else:
            gen_seq = sampler.sampling(model, z, batch)
        # reshape the batch from a seq to a matrix
        gen_seqs = post_process(gen_seq, inverse_scaler)
        return gen_seqs

    return sampling_fn


def get_partial_sampling_fn(config, noise_scheduler, sampling_steps, inverse_scaler, eps=1e-3):
    device = config.device

    time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
    sampler = AncestralSampler(config, noise_scheduler, time_steps, config.model.pred_data, inverse_scaler)

    @torch.no_grad()
    def sampling_fn(model, data, fixed_part=True):
        assert fixed_part
        model.eval()
        # extend the sequence into a batch according to n_samples
        batch = data
        # sample initial noise
        gen_seq = sampler.partial_sampling_fixed(model, batch)
        # reshape the batch from a seq to a matrix
        gen_seqs = post_process(gen_seq, inverse_scaler)
        return gen_seqs

    return sampling_fn


def get_multi_state_sampling_fn(config, noise_scheduler, sampling_steps, inverse_scaler, eps=1e-3):
    device = config.device

    time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
    sampler = AncestralSampler(config, noise_scheduler, time_steps, config.model.pred_data, inverse_scaler)

    @torch.no_grad()
    def sampling_fn(model, data, fixed_part=False):
        model.eval()
        # extend the sequence into a batch according to n_samples
        batch = data
        # sample initial noise
        z = torch.randn(batch['node_s'].shape, device=device)
        if fixed_part:
            gen_seq = sampler.multi_sampling_fixed(model, z, batch)
        else:
            gen_seq = sampler.multi_sampling(model, z, batch)
        # reshape the batch from a seq to a matrix
        gen_seqs = post_process(gen_seq, inverse_scaler)
        return gen_seqs

    return sampling_fn


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class AncestralSampler:
    """Ancestral sampling for RNA inverse design."""
    def __init__(self, config, noise_scheduler, time_steps, model_pred_data, inverse_scaler):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.self_cond = config.model.self_cond
        self.partial_min_t = config.eval.partial_min_t
        self.partial_max_t = config.eval.partial_max_t
        self.cond_scale = getattr(config.eval, 'cond_scale', -1.)
        self.dynamic_threshold = getattr(config.eval, 'dynamic_threshold', False)
        self.dynamic_thresholding_percentile = getattr(config.eval, 'dynamic_thresholding_percentile', 0.95)

    @torch.no_grad()
    def forward_with_cond_scale(self, model, *args, cond_scale=1.0, **kwargs):
        logits = model(*args, **kwargs)  # with condition
        if cond_scale == 1.0:
            return logits

        null_logits = model(*args, cond_drop_prob=1.0, **kwargs)  # without condition
        return null_logits + (logits - null_logits) * cond_scale

    @torch.no_grad()
    def sampling(self, model, z_T, batch):
        batch['z_t'] = x = z_T
        bs = z_T.shape[0]
        cond_x = None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            # vec_t = torch.ones(bs, device=x.device) * t
            # noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

            # prediction with model
            pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x) \
                if self.cond_scale < 0. \
                else self.forward_with_cond_scale(model, batch, cond_scale=self.cond_scale, time=t.unsqueeze(0),
                                                 noise_level=noise_level.unsqueeze(0), cond_x=cond_x)

            # dynamic thresholding
            if self.dynamic_threshold:
                # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
                s = torch.quantile(
                    pred_t.reshape(bs, -1).abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)

                s = expand_dims(s, pred_t.dim())
                pred_t = pred_t.clamp(-s, s) / s

            if self.self_cond:
                assert self.model_pred_data
                cond_x = pred_t.detach().clone()

            # seq update
            if pred_t.shape != x.shape:
                pred_t = pred_t.unsqueeze(-2)

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            batch['z_t'] = x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                torch.randn(x_mean.shape, device=x.device)
                # torch.randn((bs, x_mean.shape[1], x_mean.shape[2] - 3), device=x.device)
        return x_mean.squeeze(-2)

    @torch.no_grad()
    def multi_sampling(self, model, z_T, batch):
        batch['z_t'] = x = z_T
        bs = z_T.shape[0]
        cond_x = None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            # noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

            if self.self_cond:
                assert self.model_pred_data
                pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x)
                cond_x = pred_t.detach().clone()
            else:
                pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x)

            # mean along batch
            pred_t = torch.mean(pred_t, dim=0, keepdim=True).repeat(bs, 1, 1)

            # seq update
            if pred_t.shape != x.shape:
                pred_t = pred_t.unsqueeze(-2)

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            # mean along the batch dimension for multi-state design
            # x_mean = torch.mean(x_mean, dim=0, keepdim=True).repeat(bs, 1, 1)

            batch['z_t'] = x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                               torch.randn(x_mean.shape, device=x.device)
            # torch.randn((bs, x_mean.shape[1], x_mean.shape[2] - 3), device=x.device)
        return x_mean.squeeze(-2)

    @torch.no_grad()
    def multi_sampling_fixed(self, model, z_T, batch):
        # Add fixed_mask to sampling function
        fixed_mask = batch['fixed_mask'].unsqueeze(-1).float()
        scale_node_s = batch['node_s'] * 2. - 1.

        # replace the fixed part in z_T with the fixed part in batch['node_s']
        z_T = scale_node_s * fixed_mask + z_T * (1 - fixed_mask)

        batch['z_t'] = x = z_T
        bs = z_T.shape[0]
        cond_x = None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            # noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

            if self.self_cond:
                assert self.model_pred_data
                pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x)
                cond_x = pred_t.detach().clone()
            else:
                pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x)

            # mean along batch
            pred_t = torch.mean(pred_t, dim=0, keepdim=True).repeat(bs, 1, 1)

            # seq update
            if pred_t.shape != x.shape:
                pred_t = pred_t.unsqueeze(-2)

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                torch.randn(x_mean.shape, device=x.device)
            x = scale_node_s * fixed_mask + x * (1 - fixed_mask)
            batch['z_t'] = x

        x_mean = scale_node_s * fixed_mask + x_mean * (1 - fixed_mask)
        return x_mean.squeeze(-2)

    @torch.no_grad()
    def sampling_fixed(self, model, z_T, batch):
        # Add fixed mask to sampling function
        fixed_mask = batch['fixed_mask'].unsqueeze(-1).float()
        scale_node_s = batch['node_s'] * 2. - 1.

        # replace the fixed part in z_T with the fixed part in batch['node_s']
        z_T = scale_node_s * fixed_mask + z_T * (1 - fixed_mask)

        batch['z_t'] = x = z_T
        bs = z_T.shape[0]
        cond_x = None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            # vec_t = torch.ones(bs, device=x.device) * t
            # noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

            # prediction with model
            pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x) \
                if self.cond_scale < 0. \
                else self.forward_with_cond_scale(model, batch, cond_scale=self.cond_scale, time=t.unsqueeze(0),
                                                 noise_level=noise_level.unsqueeze(0), cond_x=cond_x)

            # dynamic thresholding
            if self.dynamic_threshold:
                # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
                s = torch.quantile(
                    pred_t.reshape(bs, -1).abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)

                s = expand_dims(s, pred_t.dim())
                pred_t = pred_t.clamp(-s, s) / s

            if self.self_cond:
                assert self.model_pred_data
                cond_x = pred_t.detach().clone()

            # seq update
            if pred_t.shape != x.shape:
                pred_t = pred_t.unsqueeze(-2)

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            # replace the fixed part in x_mean with the fixed part in batch['node_s']
            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                torch.randn(x_mean.shape, device=x.device)
            x = scale_node_s * fixed_mask + x * (1 - fixed_mask)
            batch['z_t'] = x

        x_mean = scale_node_s * fixed_mask + x_mean * (1 - fixed_mask)
        return x_mean.squeeze(-2)

    @torch.no_grad()
    def partial_sampling_fixed(self, model, batch):
        # Add fixed mask to sampling function, and process partial sampling
        fixed_mask = batch['fixed_mask'].unsqueeze(-1).float()
        scale_node_s = batch['node_s'] * 2. - 1.

        # add partial noise to sequence
        # sample t between 0.15 and 0.5
        t_random = (torch.rand(1, device=scale_node_s.device) * (self.partial_max_t - self.partial_min_t) +
                    self.partial_min_t)
        # find the closest t in t_array
        t_partial = self.t_array[torch.argmin(torch.abs(self.t_array - t_random))]
        idx_partial = torch.where(self.t_array == t_partial)[0]

        # sample the noisy sample at t_partial
        alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t_partial)
        noise = torch.randn(scale_node_s.shape, device=scale_node_s.device)
        z_t = expand_dims(alpha_t, scale_node_s.dim()) * scale_node_s + expand_dims(sigma_t, noise.dim()) * noise
        z_t = scale_node_s * fixed_mask + z_t * (1 - fixed_mask)

        batch['z_t'] = x = z_t
        bs = z_t.shape[0]
        cond_x = None
        for i in range(idx_partial, len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            # vec_t = torch.ones(bs, device=x.device) * t
            noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

            # prediction with model
            pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x) \
                if self.cond_scale < 0. \
                else self.forward_with_cond_scale(model, batch, cond_scale=self.cond_scale, time=t.unsqueeze(0),
                                                 noise_level=noise_level.unsqueeze(0), cond_x=cond_x)

            # dynamic thresholding
            if self.dynamic_threshold:
                # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
                s = torch.quantile(
                    pred_t.reshape(bs, -1).abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)

                s = expand_dims(s, pred_t.dim())
                pred_t = pred_t.clamp(-s, s) / s

            if self.self_cond:
                assert self.model_pred_data
                cond_x = pred_t.detach().clone()

            # seq update
            if pred_t.shape != x.shape:
                pred_t = pred_t.unsqueeze(-2)

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            # replace the fixed part in x_mean with the fixed part in batch['node_s']
            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                torch.randn(x_mean.shape, device=x.device)
            x = scale_node_s * fixed_mask + x * (1 - fixed_mask)
            batch['z_t'] = x

        x_mean = scale_node_s * fixed_mask + x_mean * (1 - fixed_mask)
        return x_mean.squeeze(-2)