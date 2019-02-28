# This Pix2Pix implementation is based on
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import lr_scheduler
import functools
from tensorboardX import SummaryWriter
from data import DataIterator

from models.base import BaseModel, generate_noise
from constants import MODE_BASE, MODE_PRED, MODE_MR


class Pix2Pix(BaseModel):
    name = 'pix2pix'

    def __init__(self, config):
        # Define D & G
        if config.mode in [MODE_BASE, MODE_MR]:
            net_d = define_d(
                config.model.in_channels + config.model.out_channels,
                ndf=config.model.num_features,
                which_model_net_d='basic',
                use_sigmoid=False
            )
            net_g = Pix2PixUnet(config.mode, config)
        else:
            net_d, net_g = None, None

        # Define P
        if config.mode in [MODE_PRED, MODE_MR] and config.model.predictor:
            net_p = Pix2PixUnet(MODE_PRED, config)
        else:
            net_p = None

        super().__init__(config, net_d, net_g, net_p)

    def optimize_p(self, x, y, step, summarize=False):
        self.optim_p.zero_grad()
        scalar_summary = {}

        # Calculate losses
        mean, log_var = self.net_p(x)
        loss = self.mle_loss(mean, log_var, y)
        loss.backward()

        self.clip_grad(self.optim_p, self.config.p_optimizer.clip_grad)
        self.optim_p.step()

        # Summarize
        if summarize:
            scalar_summary['loss/p/mle'] = loss.detach()
            summary = {'scalar': scalar_summary}
            return summary

    def optimize_d(self, x, y, step, summarize=False):
        torch.cuda.empty_cache()
        if self.loss_config.gan_weight == 0.:
            return {}

        self.optim_d.zero_grad()
        with torch.no_grad():
            fake_y = self(x)
        fake_v = self.net_d(torch.cat([x, fake_y], dim=1))
        real_v = self.net_d(torch.cat([x, y], dim=1))
        real_loss = self.gan_loss(real_v, True)
        fake_loss = self.gan_loss(fake_v, False)
        loss = (real_loss + fake_loss) * 0.5 * self.loss_config.gan_weight

        loss.backward()
        self.clip_grad(self.optim_d, self.config.d_optimizer.clip_grad)
        self.optim_d.step()

        if summarize:
            scalar_summary = {
                'loss/d/gan_real': real_loss,
                'loss/d/gan_fake': fake_loss,
            }
            return {'scalar': scalar_summary}

    def mle_target(self, y):
        return y

    def build_d_input(self, x, samples):
        num_mr = self.config.num_mr
        num_samples = self.config.num_mr_samples

        x_dup = x[:num_mr].unsqueeze(1)
        x_dup = x_dup.expand(-1, num_samples, -1, -1, -1)
        x_dup = x_dup.contiguous().view(
            num_mr * num_samples, *list(x_dup.size()[2:]))
        return torch.cat([x_dup, samples], 1)

    def optimize_g(self, x, y, step, summarize=False):
        torch.cuda.empty_cache()
        scalar = {'loss/g/total': 0.}
        histogram = {}
        image = {}
        loss = 0.
        self.optim_g.zero_grad()

        if self.mode == MODE_BASE or self.loss_config.recon_weight > 0:
            fake_y, _ = self.net_g(x)

            # GAN loss
            if self.loss_config.gan_weight > 0:
                fake_v = self.net_d(torch.cat([x, fake_y], dim=1))
                gan_loss = self.gan_loss(fake_v, True)
                weighted_gan_loss = self.loss_config.gan_weight * gan_loss
                loss += weighted_gan_loss
                if summarize:
                    scalar['loss/g/gan'] = gan_loss.detach()
                    scalar['loss/g/total'] += weighted_gan_loss.detach()

            # L1 loss
            if self.loss_config.recon_weight > 0:
                l1_loss = self._l1_loss(fake_y, y)
                weighted_l1_loss = self.loss_config.recon_weight * l1_loss
                loss += weighted_l1_loss
                if summarize:
                    scalar['loss/g/l1'] = l1_loss.detach()
                    scalar['loss/g/total'] += weighted_l1_loss.detach()

            # Back-prop
            loss.backward()

        # Moment matching loss
        if self.mode == MODE_MR and (
                self.loss_config.mr_1st_weight > 0
                or self.loss_config.mr_2nd_weight > 0
                or self.loss_config.mle_weight > 0
        ):
            mr_summaries = self.accumulate_mr_grad(x, y, summarize)
            mr_scalar = mr_summaries['scalar']
            mr_histogram = mr_summaries['histogram']
            mr_image = mr_summaries['image']
            torch.cuda.empty_cache()

            if summarize:
                scalar['loss/g/total'] += mr_scalar['loss/g/total']
                del mr_scalar['loss/g/total']
                scalar.update(mr_scalar)
                histogram.update(mr_histogram)
                for i in range(min(16, self.config.num_mr)):
                    image_id = 'train_samples/%d' % i
                    if image_id in image:
                        image[image_id] = torch.cat([
                            image[image_id], mr_image[image_id]
                        ], 2)
                    else:
                        image[image_id] = mr_image[image_id]
                    image[image_id] = image[image_id] * 0.5 + 0.5

        # Optimize
        self.clip_grad(self.optim_g, self.config.g_optimizer.clip_grad)
        self.optim_g.step()

        return {'scalar': scalar, 'histogram': histogram, 'image': image}

    def summarize(self, writer: SummaryWriter, step,
                  train_sample_iterator: DataIterator,
                  val_sample_iterator: DataIterator):
        for iter_type, iterator in (
                ('train', train_sample_iterator), ('val', val_sample_iterator)):
            mle_loss = 0.
            for i in range(len(iterator)):
                x, y = next(iterator)
                if self.mode == MODE_BASE:
                    with torch.no_grad():
                        self.net_g.eval()
                        fake_y = self.net_g(x)[0]
                        self.net_g.train()
                        x_plain = self.undo_norm(x)
                        collage = torch.cat([x_plain, y, fake_y], dim=3)
                        collage = 0.5 * collage + 0.5
                    writer.add_image('g/%s/%d' % (iter_type, i), collage, step)

                elif self.mode == MODE_MR:
                    with torch.no_grad():
                        pred_1st, pred_log_2nd = self.net_p(x)
                        self.net_g.eval()
                        num_samples = 12
                        g_x = x.expand(num_samples, -1, -1, -1)
                        samples = self.net_g(g_x)[0]
                        self.net_g.train()

                    # Build collage
                    sample_1st, sample_2nd = self.sample_statistics(samples)
                    log_2nds = torch.cat([
                        pred_log_2nd, (sample_2nd + 1e-10).log()
                    ], 3)
                    uncertainty = self._build_uncertainty_image(log_2nds)
                    x_plain = self.undo_norm(x)
                    collage = [
                        torch.cat([
                            x_plain, y, pred_1st, sample_1st, uncertainty,
                        ], dim=3),
                        torch.cat([
                            fy.unsqueeze(0)
                            for fy in torch.unbind(samples[:num_samples // 2])
                        ], dim=3),
                        torch.cat([
                            fy.unsqueeze(0)
                            for fy in torch.unbind(samples[num_samples // 2:])
                        ], dim=3)
                    ]
                    collage = torch.cat(collage, dim=2)
                    collage = collage * 0.5 + 0.5
                    writer.add_image('collage/%s/%d' % (iter_type, i), collage,
                                     step)

                elif self.mode == MODE_PRED:
                    with torch.no_grad():
                        self.net_p.eval()
                        pred_1st, pred_log_2nd = self.net_p(x)
                        self.net_p.train()

                        uncertainty = self._build_uncertainty_image(
                            pred_log_2nd)

                        x_plain = self.undo_norm(x)
                        collage = torch.cat([
                            torch.cat([x_plain, y], dim=3),
                            torch.cat([pred_1st, uncertainty], dim=3)
                        ], dim=2)
                        collage = collage * 0.5 + 0.5
                    writer.add_image('p/%s/%d' % (iter_type, i), collage, step)

                    # Validation loss
                    if iter_type == 'val':
                        mle_loss += self.mle_loss(pred_1st, pred_log_2nd, y)

            if self.mode == MODE_PRED and iter_type == 'val':
                writer.add_scalar('loss/p/mle_val', mle_loss / len(iterator),
                                  step)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                       track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_d(input_nc, ndf, which_model_net_d,
             n_layers_d=3, norm='batch', use_sigmoid=False, init_type='normal',
             init_gain=0.02, gpu_ids=()):
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_net_d == 'basic':
        net_d = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer,
            use_sigmoid=use_sigmoid)
    elif which_model_net_d == 'n_layers':
        net_d = NLayerDiscriminator(
            input_nc, ndf, n_layers_d, norm_layer=norm_layer,
            use_sigmoid=use_sigmoid)
    elif which_model_net_d == 'pixel':
        net_d = PixelDiscriminator(
            input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' %
            which_model_net_d)
    return init_net(net_d, init_type, init_gain, gpu_ids)


class Pix2PixUnet(nn.Module):
    def __init__(self, mode, config):
        super().__init__()
        self.mode = mode

        # Config shortcuts
        self.in_ch = config.model.in_channels
        self.out_ch = config.model.out_channels
        self.num_downs = config.model.num_downs
        if self.mode == MODE_PRED:
            self.num_features = config.model.pred_features
        else:
            self.num_features = config.model.num_features
        self.noise_type = config.model.noise_type
        self.noise_dim = config.model.noise_dim

        # Setup normalization layer
        if config.model.norm == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif config.model.norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        else:
            raise ValueError('Invalid norm layer type {}'
                             .format(config.model.norm))

        down_convs = []
        up_convs = []
        down_norms = []
        up_norms = []

        # Create down conv
        prev_ch = self.in_ch
        next_ch = self.num_features
        for i in range(self.num_downs):
            down_conv = nn.Conv2d(prev_ch, next_ch, kernel_size=4,
                                  stride=2, padding=1, bias=i == 0)
            down_convs.append(down_conv)
            if i != 0:
                down_norm = norm_layer(next_ch)
                down_norms.append(down_norm)
            else:
                down_norms.append(None)

            prev_ch = next_ch
            next_ch = min(2 * next_ch, self.num_features * 8)

        self.down_convs = nn.ModuleList(down_convs)
        self.down_norms = nn.ModuleList(down_norms)

        # Create up conv in reverse order
        prev_ch = self.out_ch
        next_ch = self.num_features
        self.mr_ch, self.up_ch = [], []
        for i in range(self.num_downs):
            if self.mode == MODE_MR:
                noise_dim = self.noise_dim[i + 1]
            else:
                noise_dim = 0
            if i == self.num_downs - 1:
                ch = next_ch
            else:
                ch = 2 * next_ch  # 2x for skip connection

            # Up convolution
            self.up_ch.append(prev_ch)
            up_conv = nn.ConvTranspose2d(
                ch + noise_dim, self.up_ch[-1],
                kernel_size=4, stride=2, padding=1, bias=i == 0 or False)
            up_convs.append(up_conv)
            if i != 0:
                up_norm = norm_layer(self.up_ch[-1])
                up_norms.append(up_norm)
            else:
                up_norms.append(None)

            prev_ch = next_ch
            next_ch = min(2 * next_ch, self.num_features * 8)

        self.up_convs = nn.ModuleList(up_convs)
        self.up_norms = nn.ModuleList(up_norms)

        # Variance prediction
        if self.mode == MODE_PRED:
            self.dispersion_conv = nn.ConvTranspose2d(
                self.num_features * 2, self.out_ch,
                kernel_size=4, stride=2, padding=1
            )
        else:
            self.dispersion_conv = None

    def forward(self, x, num_samples=1):
        # Down conv
        feat = [x]
        h = x
        for i, down_conv in enumerate(self.down_convs):
            f = down_conv(h)
            h = F.leaky_relu(f, 0.2)
            if self.down_norms[i] is not None:
                h = self.down_norms[i](h)
                feat.append(h)
            else:
                feat.append(h)

        # Duplicate num_samples times
        if num_samples > 1:
            feat = [
                f.unsqueeze(1).expand(
                    -1, num_samples, -1, -1, -1
                ).contiguous().view(
                    f.size(0) * num_samples, f.size(1), f.size(2), f.size(3))
                for f in feat
            ]
            h = h.unsqueeze(1).expand(-1, num_samples, -1, -1, -1).contiguous()
            h = h.view(h.size(0) * h.size(1), h.size(2), h.size(3), h.size(4))

        # Up conv & MR conv
        dispersion = None
        for i in range(len(self.up_convs))[::-1]:
            # Skip connection
            if i < len(self.up_convs) - 1:
                h = torch.cat([h, feat[i + 1]], dim=1)

            # Predict dispersion
            if self.mode == MODE_PRED and i == 0:
                dispersion = self.dispersion_conv(h)

            # Mix noise
            if self.mode == MODE_MR and self.noise_dim[i + 1] > 0:
                z = generate_noise(self.noise_type, self.noise_dim[i + 1],
                                   like=h)
                h = torch.cat([h, z], dim=1)

            # Up convolution
            h = self.up_convs[i](h)
            h = torch.tanh(h) if i == 0 else F.relu(h)
            if self.up_norms[i] is not None:
                h = self.up_norms[i](h)

        return h, dispersion


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0,
                      bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0,
                      bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
