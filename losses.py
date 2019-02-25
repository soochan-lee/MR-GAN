import torch
from torch import nn


class L1Loss(nn.Module):
    name = 'l1'

    def __init__(self):
        super().__init__()

    def forward(self, x, y, normalizer=None):
        if not normalizer:
            return (x - y).abs().mean()
        else:
            return (x - y).abs().sum() / normalizer


class MSELoss(nn.Module):
    name = 'mse'

    def __init__(self):
        super().__init__()

    def forward(self, x, y, normalizer=None):
        if not normalizer:
            return ((x - y) ** 2).mean()
        else:
            return ((x - y) ** 2).sum() / normalizer


class GANLoss(nn.Module):
    name = 'gan'
    need_lipschitz_d = False

    def __init__(self, real_label=1.0, fake_label=0.0, with_logits=False):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if with_logits:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.BCELoss()

    def forward(self, verdict, target_is_real):
        if target_is_real:
            target = self.real_label
        else:
            target = self.fake_label
        return self.loss(verdict, target.expand_as(verdict))


class WGANLoss(nn.Module):
    name = 'wgan'
    need_lipschitz_d = True

    def __init__(self, with_logits=True):
        super().__init__()

    def forward(self, verdict, target_is_real):
        return -verdict.mean() if target_is_real else verdict.mean()


class LSGANLoss(nn.Module):
    name = 'lsgan'
    need_lipschitz_d = False

    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.MSELoss()

    def forward(self, verdict, target_is_real):
        if target_is_real:
            target = self.real_label
        else:
            target = self.fake_label
        return self.loss(verdict, target.expand_as(verdict))


class GaussianMLELoss(nn.Module):
    name = 'gaussian'

    def __init__(self, order=2, min_noise=0.):
        super().__init__()
        self.order = order
        self.min_noise = min_noise

    def forward(self, center, dispersion, y,
                log_dispersion=True, normalizer=None):
        squared = (center - y) ** 2

        if self.order == 1:
            return squared.mean()

        if log_dispersion:
            var = dispersion.exp()
            log_var = dispersion
        else:
            var = dispersion
            log_var = (dispersion + 1e-9).log()

        loss = ((squared + self.min_noise) / (var + 1e-9) + log_var) * 0.5

        if not normalizer:
            return loss.mean()
        else:
            return loss.sum() / normalizer


class LaplaceMLELoss(nn.Module):
    name = 'laplace'

    def __init__(self, order=2, min_noise=0.):
        super().__init__()
        self.order = order
        self.min_noise = min_noise

    def forward(self, center, dispersion, y,
                log_dispersion=True, normalizer=None):
        deviation = (center - y).abs()

        if self.order == 1:
            return deviation.mean()

        if log_dispersion:
            mad = dispersion.exp()
            log_mad = dispersion
        else:
            mad = dispersion
            log_mad = (dispersion + 1e-9).log()

        loss = (deviation + self.min_noise) / (mad + 1e-9) + log_mad

        if not normalizer:
            return loss.mean()
        else:
            return loss.sum() / normalizer


LOSSES = {
    GANLoss.name: GANLoss,
    WGANLoss.name: WGANLoss,
    LSGANLoss.name: LSGANLoss,
    GaussianMLELoss.name: GaussianMLELoss,
    LaplaceMLELoss.name: LaplaceMLELoss,
}
