import math
import torch
from torch import nn
from data import DataIterator
import os
from tensorboardX import SummaryWriter
from constants import MODES, MODE_BASE, MODE_PRED, MODE_MR
from losses import LOSSES, LaplaceMLELoss, GaussianMLELoss, L1Loss, MSELoss


class BaseModel(nn.Module):
    def __init__(self, config,
                 net_d: nn.Module = None, net_g: nn.Module = None,
                 net_p: nn.Module = None):
        super().__init__()
        self.config = config
        assert config.mode in MODES
        self.mode = config.mode
        self.log_dir = config.log_dir
        self.loss_config = getattr(config.model.losses, self.mode)
        if getattr(config.data, 'norm', None) is not None:
            self.input_mean = torch.tensor(
                config.data.norm.mean, device='cuda'
            ).view(1, 3, 1, 1)
            self.input_std = torch.tensor(
                config.data.norm.std, device='cuda'
            ).view(1, 3, 1, 1)
        else:
            self.input_mean = None
            self.input_std = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu')
        self.net_d = net_d
        self.net_g = net_g
        self.net_p = net_p

        # GAN loss
        self.gan_loss = LOSSES[config.model.gan.type](
            with_logits=config.model.gan.with_logits
        )
        if self.net_d is not None and self.gan_loss.need_lipschitz_d:
            for m in [m for m in self.net_d.modules() if m._parameters]:
                nn.utils.spectral_norm(m)

        # MLE loss
        self.mle = config.model.mle.type
        if self.mle == 'gaussian':
            self.mle_loss = GaussianMLELoss(
                **config.model.mle.options._asdict())
        elif self.mle == 'laplace':
            self.mle_loss = LaplaceMLELoss(
                **config.model.mle.options._asdict())
        else:
            raise ValueError('Invalid MLE loss type: %s' % self.mle)

        # Other losses
        self._l1_loss = L1Loss()
        self._mse_loss = MSELoss()

        # Build optimizers
        self.optim_d, self.optim_g, self.optim_p = None, None, None
        if net_d is not None and self.mode != MODE_PRED \
                and self.loss_config.gan_weight:
            self.optim_d = self._build_optimizer(
                config.d_optimizer, net_d.parameters())
        if net_g is not None and self.mode != MODE_PRED:
            self.optim_g = self._build_optimizer(
                config.g_optimizer, net_g.parameters())
        if net_p is not None and self.mode == MODE_PRED:
            self.optim_p = self._build_optimizer(
                config.p_optimizer, net_p.parameters())

        # Build learning rate schedulers
        self.lr_sched_d, self.lr_sched_g, self.lr_sched_p = None, None, None
        if self.optim_d is not None and config.d_lr_scheduler is not None:
            self.lr_sched_d = self._build_lr_scheduler(
                config.d_lr_scheduler, self.optim_d)
        if self.optim_g is not None and config.g_lr_scheduler is not None:
            self.lr_sched_g = self._build_lr_scheduler(
                config.g_lr_scheduler, self.optim_g)
        if self.optim_p is not None and config.p_lr_scheduler is not None:
            self.lr_sched_p = self._build_lr_scheduler(
                config.p_lr_scheduler, self.optim_p)

        # Set train/eval
        if self.mode == MODE_BASE:
            net_g.train()
            net_d.train()
        elif self.mode == MODE_PRED:
            net_p.train()
        elif self.mode == MODE_MR:
            net_g.train()
            net_d.train()
            if net_p is not None:
                net_p.eval()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.mode == MODE_PRED:
            return self.net_p(x)[0]
        elif self.mode == MODE_MR:
            return self.net_g(x)[0]
        else:
            return self.net_g(x)[0]

    def optimize_d(self, x, y, step, summarize=False):
        raise NotImplementedError

    def optimize_g(self, x, y, step, summarize=False):
        raise NotImplementedError

    def optimize_p(self, x, y, step, summarize=False):
        raise NotImplementedError

    def summarize(self, writer: SummaryWriter, step,
                  train_sample_iterator: DataIterator,
                  val_sample_iterator: DataIterator):
        raise NotImplementedError

    @staticmethod
    def write_summary(writer: SummaryWriter, summary, step):
        for summary_type, summary_fn in [
            ('scalar', writer.add_scalar),
            ('image', writer.add_image),
            ('histogram', writer.add_histogram),
            ('text', writer.add_text)
        ]:
            if summary_type not in summary:
                continue
            for name, value in summary[summary_type].items():
                summary_fn(name, value, step)

    def save(self, step):
        os.makedirs(os.path.join(self.log_dir, 'ckpt'), exist_ok=True)
        if self.net_d is not None:
            save_path = os.path.join(self.log_dir, 'ckpt', '%d-d.pt' % step)
            torch.save({'state': self.net_d.state_dict(), 'step': step},
                       save_path)
        if self.net_g is not None:
            save_path = os.path.join(self.log_dir, 'ckpt', '%d-g.pt' % step)
            torch.save({'state': self.net_g.state_dict(), 'step': step},
                       save_path)
        if self.net_p is not None:
            save_path = os.path.join(self.log_dir, 'ckpt', '%d-p.pt' % step)
            torch.save({'state': self.net_p.state_dict(), 'step': step},
                       save_path)

    def load(self, path):
        step = 0
        if self.net_d is not None:
            step = self.load_module(self.net_d, path + '-d.pt')
        if self.net_g is not None:
            step = self.load_module(self.net_g, path + '-g.pt')
        if self.net_p is not None:
            step = self.load_module(self.net_p, path + '-p.pt')
        return step

    def _build_uncertainty_image(self, log_var):
        log_var = log_var.clone()
        log_var_min = self.config.log_dispersion_min
        log_var_gap = self.config.log_dispersion_max - log_var_min

        # Add legend
        h = log_var.size(2)
        l_size = min(h // 10, 10)
        for i in range(8):
            log_var[:, :, i * l_size:(i + 1) * l_size, 0:l_size] = i * -1.
        uncertainty = log_var - log_var_min
        uncertainty /= (log_var_gap * 0.5)
        uncertainty -= 1.0
        uncertainty = torch.clamp(uncertainty, min=-1., max=1.)
        return uncertainty.expand([-1, 3, -1, -1])

    @staticmethod
    def _build_sigma_offset_images(decode_fn, mean, log_var, index):
        stddev = log_var.exp().sqrt()
        mean_noffset = mean.clone()
        mean_noffset[:, :stddev.size(1), ...] -= stddev
        mean_poffset = mean.clone()
        mean_poffset[:, :stddev.size(1), ...] += stddev
        noffset = decode_fn(mean_noffset, index)
        poffset = decode_fn(mean_poffset, index)
        return noffset.clamp(-1, 1), poffset.clamp(-1, 1)

    @staticmethod
    def load_module(module: nn.Module, path, strict=True):
        ckpt = torch.load(path)
        module.load_state_dict(ckpt['state'], strict=strict)
        return ckpt['step']

    @staticmethod
    def _build_optimizer(optim_config, params) -> torch.optim.Optimizer:
        return getattr(torch.optim, optim_config.type)(
            params, **optim_config.options._asdict())

    @staticmethod
    def _build_lr_scheduler(lr_config, optimizer):
        return getattr(torch.optim.lr_scheduler, lr_config.type)(
            optimizer, **lr_config.options._asdict())

    @staticmethod
    def _clip_grad_value(optimizer, clip_value):
        for group in optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    @staticmethod
    def _clip_grad_norm(optimizer, max_norm, norm_type=2):
        for group in optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    @staticmethod
    def clip_grad(optimizer, clip_grad_config=None):
        if clip_grad_config is None:
            return

        if clip_grad_config.type == 'value':
            BaseModel._clip_grad_value(
                optimizer, **clip_grad_config.options._asdict()
            )
        elif clip_grad_config.type == 'norm':
            BaseModel._clip_grad_norm(
                optimizer, **clip_grad_config.options._asdict()
            )
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config.type))

    def undo_norm(self, x):
        if self.input_mean is None:
            # [-1, 1] -> [-1, 1]
            return x
        else:
            # N(0, 1) -> [-1, 1]
            return (x * self.input_std + self.input_mean) * 2. - 1.

    def sample_statistics(self, samples):
        """Calculate sample statistics

        Args:
            samples: 5D tensor of shape <B, S, C, H, W>
                or 4D tensor of shape <S, C, H, W>
        Returns:
            sample_1st: 4D tensor of shape <B, C, H, W>
            sample_2nd: 4D tensor of shape <B, ?, H, W>
        """
        if len(samples.size()) == 4:
            samples = samples.unsqueeze(0)
        if isinstance(self.mle_loss, GaussianMLELoss):
            num_samples = samples.size(1)
            sample_1st = samples.mean(dim=1, keepdim=True)
            # Tensor.std has bug up to PyTorch 0.4.1
            sample_2nd = (samples - sample_1st) ** 2
            sample_2nd = sample_2nd.sum(dim=1, keepdim=True)
            sample_2nd /= num_samples - 1

        # Laplace statistics
        elif isinstance(self.mle_loss, LaplaceMLELoss):
            sample_1st, _ = samples.median(
                dim=1, keepdim=True)
            sample_2nd = torch.abs(samples - sample_1st)
            sample_2nd = sample_2nd.mean(dim=1, keepdim=True)
        else:
            raise RuntimeError('Unknown type of MLE loss')

        return sample_1st.squeeze(1), sample_2nd.squeeze(1)

    def mle_target(self, y):
        """Process ground truth y to proper MLE target"""
        raise NotImplementedError

    def build_d_input(self, x, samples):
        """Build discriminator input"""
        raise NotImplementedError

    def accumulate_mm_grad(self, x, y, summarize=False):
        # Initialize summaries
        scalar = {}
        histogram = {}
        image = {}

        num_mm = self.config.num_mm
        num_samples = self.config.num_mm_samples

        loss = 0.

        # Get predictive mean and variance
        if self.net_p is not None:
            with torch.no_grad():
                pred_1st, pred_log_2nd = self.net_p(x[:num_mm])
                pred_2nd = torch.exp(pred_log_2nd)

            if summarize:
                scalar['variance/pred'] = pred_2nd.detach().mean()
                histogram['variance_hist/pred'] = pred_2nd.detach().view(-1)
        else:
            pred_1st, pred_log_2nd, pred_2nd = None, None, None

        # Get samples
        samples, _ = self.net_g(x[:num_mm], num_samples=num_samples)

        # GAN loss
        if self.loss_config.gan_weight > 0:
            d_input = self.build_d_input(x, samples)
            fake_v = self.net_d(d_input)
            gan_loss = self.gan_loss(fake_v, True)
            loss += self.loss_config.gan_weight * gan_loss

            if summarize:
                scalar['loss/g/gan'] = gan_loss.detach()

        # Get sample mean and variance
        samples = samples.view(
            num_mm, num_samples, *list(samples.size()[1:]))
        sample_1st, sample_2nd = self.sample_statistics(samples)

        if summarize:
            for i in range(min(16, samples.size(0))):
                image['train_samples/%d' % i] = torch.cat(
                    torch.unbind(samples[i, :5].detach()), 2
                )
            scalar['variance/sample'] = sample_2nd.detach().mean()
            histogram['variance_hist/sample'] = sample_2nd.detach().view(-1)

        # Direct MLE without predictor
        if self.loss_config.mle_weight > 0:
            if self.name == 'glcic':
                masks = x[:num_mm, -1:, ...]
                sample_2nd = (
                    sample_2nd * masks +
                    math.exp(self.config.log_dispersion_min) * (1. - masks)
                )
                normalizer = x[:num_mm, -1:, ...].sum()
            else:
                normalizer = None
            if isinstance(self.mle_loss, GaussianMLELoss):
                mle_loss = self.mle_loss(
                    sample_1st, sample_2nd, self.mle_target(y[:num_mm]),
                    log_dispersion=False, normalizer=normalizer)
            elif isinstance(self.mle_loss, LaplaceMLELoss):
                sample_mean = samples.mean(1)
                with torch.no_grad():
                    deviation = self.mle_target(y[:num_mm]) - sample_1st
                    mean_target = (deviation + sample_mean).detach()
                mle_loss = self.mle_loss(
                    sample_mean, sample_2nd, mean_target,
                    log_dispersion=False, normalizer=normalizer
                )
            else:
                raise RuntimeError('Invalid MLE loss')

            loss += self.loss_config.mle_weight * mle_loss

            if summarize:
                scalar['loss/g/mle'] = mle_loss.detach()

        # Moment matching
        if self.loss_config.mm_1st_weight or self.loss_config.mm_2nd_weight:
            normalizer = (
                x[:num_mm, -1:, ...].sum() if self.name == 'glcic' else None
            )
            if isinstance(self.mle_loss, GaussianMLELoss):
                mm_1st_loss = self._mse_loss(
                    sample_1st, pred_1st, normalizer=normalizer
                )
            elif isinstance(self.mle_loss, LaplaceMLELoss):
                sample_mean = samples.mean(1)
                with torch.no_grad():
                    mean_target = (pred_1st - sample_1st + sample_mean).detach()
                mm_1st_loss = self._mse_loss(
                    sample_mean, mean_target, normalizer=normalizer
                )
            else:
                raise RuntimeError('Invalid MLE loss')
            mm_2nd_loss = self._mse_loss(
                sample_2nd, pred_2nd, normalizer=normalizer
            )
            weighted_mm_loss = \
                self.loss_config.mm_1st_weight * mm_1st_loss + \
                self.loss_config.mm_2nd_weight * mm_2nd_loss
            loss += weighted_mm_loss

            if summarize:
                scalar['loss/g/mm_1st'] = mm_1st_loss.detach()
                scalar['loss/g/mm_2nd'] = mm_2nd_loss.detach()
                scalar['loss/g/mm'] = weighted_mm_loss.detach()

        if summarize:
            scalar['loss/g/total'] = loss.detach()

        loss.backward()

        return {'scalar': scalar, 'histogram': histogram, 'image': image}


def generate_noise(noise_type, noise_dim, like):
    if noise_type == 'gaussian':
        z = torch.randn([
            like.size(0), noise_dim,
            like.size(2), like.size(3)
        ], device='cuda')
    elif noise_type == 'uniform':
        z = torch.rand(
            [like.size(0), noise_dim,
             like.size(2), like.size(3)],
            device='cuda'
        )
    elif noise_type == 'bernoulli':
        z = torch.bernoulli(
            torch.ones([
                like.size(0), noise_dim,
                like.size(2), like.size(3)
            ], device='cuda') * 0.5
        )
    elif noise_type == 'categorical':
        z = torch.zeros([like.size(0), noise_dim,
                         like.size(2), like.size(3)], device='cuda')
        z_idx = torch.randint(
            low=0, high=noise_dim,
            size=[like.size(0), 1, like.size(2), like.size(3)],
            dtype=torch.long, device='cuda'
        )
        z.scatter_(1, z_idx, 1)
    else:
        raise ValueError('Invalid noise type %s' % noise_type)

    return z
