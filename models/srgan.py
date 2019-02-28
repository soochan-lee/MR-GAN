# This SRGAN implementation is based on https://github.com/zijundeng/SRGAN
import math
import torch.nn.functional as F
import torch
from torch import nn
from tensorboardX import SummaryWriter
from data import DataIterator
from constants import MODE_BASE, MODE_PRED, MODE_MR
from .base import BaseModel, generate_noise


class ResidualBlock(nn.Module):
    def __init__(self, channels, noise_dim=0, noise_type=None):
        super(ResidualBlock, self).__init__()
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.conv1 = nn.Conv2d(
            channels + noise_dim, channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_in = x if self.noise_dim <= 0 else torch.cat([
            x, generate_noise(
                self.noise_type,
                self.noise_dim,
                like=x
            )
        ], 1)
        residual = self.conv1(x_in)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, noise_dim=0, noise_type=None):
        super(UpsampleBLock, self).__init__()
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.conv = nn.Conv2d(
            in_channels + noise_dim, in_channels * up_scale ** 2,
            kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = x if self.noise_dim <= 0 else torch.cat([
            x, generate_noise(
                self.noise_type,
                self.noise_dim,
                like=x
            )
        ], 1)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, config, as_predictor=False):
        self.config = config
        self.as_predictor = as_predictor
        self.noise_dim = nd = config.model.noise_dim
        self.noise_type = config.model.noise_type
        upsample_block_num = int(math.log(4, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(
            64, nd[6] if nd[6] > 0 and not as_predictor else 0, self.noise_type
        )
        self.block3 = ResidualBlock(
            64, nd[5] if nd[5] > 0 and not as_predictor else 0, self.noise_type
        )
        self.block4 = ResidualBlock(
            64, nd[4] if nd[4] > 0 and not as_predictor else 0, self.noise_type
        )
        self.block5 = ResidualBlock(
            64, nd[3] if nd[3] > 0 and not as_predictor else 0, self.noise_type
        )
        self.block6 = ResidualBlock(
            64, nd[2] if nd[2] > 0 and not as_predictor else 0, self.noise_type
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(
                64 + nd[1] if nd[1] > 0 and not as_predictor else 64, 64,
                kernel_size=3, padding=1
            ),
            nn.PReLU()
        )
        self.block8 = nn.Sequential(*[
            UpsampleBLock(
                64, 2,
                nd[0] if i == 0 and nd[0] > 0 and not as_predictor else 0,
                self.noise_type
            ) for i in range(upsample_block_num)
        ])
        self.mean_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        self.dispersion_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x, num_samples=1):
        if num_samples > 1:
            x = x\
                .unsqueeze(1)\
                .expand(-1, num_samples, -1, -1, -1)\
                .contiguous()\
                .view(x.size(0) * num_samples, x.size(1), x.size(2), x.size(3))
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block6 = block6 if \
            self.noise_dim[1] <= 0 or self.as_predictor else torch.cat([
                block6, generate_noise(
                    self.noise_type,
                    self.noise_dim[1],
                    like=block6
                )
            ], 1)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        mean = self.mean_conv(block8)
        mean = F.tanh(mean)

        if self.as_predictor:
            log_dispersion = self.dispersion_conv(block8)
            return mean, log_dispersion
        else:
            return mean, None


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class SRGAN(BaseModel):
    name = 'srgan'

    def __init__(self, config):
        super().__init__(
            config,
            Discriminator(config)
            if config.mode in (MODE_BASE, MODE_MR) else None,
            Generator(config)
            if config.mode in (MODE_BASE, MODE_MR) else None,
            Generator(config, as_predictor=True)
            if config.mode in (MODE_PRED, MODE_MR) else None,
        )

    def optimize_d(self, x, y, step, summarize=False):
        assert self.mode in (MODE_BASE, MODE_MR)
        if self.loss_config.gan_weight <= 0:
            return {}
        image = {}
        # gan loss (BASELINE || MR)
        fake_y = self.net_g(x)[0]
        fake_v = self.net_d(fake_y)
        real_v = self.net_d(y)
        real_loss = self.gan_loss(real_v, True)
        fake_loss = self.gan_loss(fake_v, False)
        gan_loss = (real_loss + fake_loss) / 2
        # backprop & minimize
        self.optim_d.zero_grad()
        loss = gan_loss * self.loss_config.gan_weight
        loss.backward()
        self.clip_grad(self.optim_d, self.config.d_optimizer.clip_grad)
        self.optim_d.step()
        # image summaries
        if summarize:
            for i in range(y.size(0)):
                real_image_id = 'd_inputs/real/{}'.format(i)
                fake_image_id = 'd_inputs/fake/{}'.format(i)
                image[real_image_id] = (y[i] + 1) / 2
                image[fake_image_id] = (fake_y[i] + 1) / 2
        return {
            'scalar': {
                'loss/d/total': loss,
                'loss/d/gan_real': real_loss,
                'loss/d/gan_fake': fake_loss,
                'loss/d/gan_total': gan_loss,
            },
            'image': image
        }

    def optimize_g(self, x, y, step, summarize=False):
        assert self.mode in (MODE_BASE, MODE_MR)
        # prepare some accumulators
        scalar = {'loss/g/total': 0.}
        histogram = {}
        image = {}
        loss = 0.
        self.optim_g.zero_grad()
        fake_y = self.net_g(x)[0]
        # GAN loss (BASELINE)
        if self.mode == MODE_BASE and self.loss_config.gan_weight > 0:
            fake_v = self.net_d(fake_y)
            gan_loss = self.gan_loss(fake_v, True)
            weighted_gan_loss = self.loss_config.gan_weight * gan_loss
            loss += weighted_gan_loss
            if summarize:
                scalar['loss/g/gan'] = gan_loss.detach()
                scalar['loss/g/total'] += weighted_gan_loss.detach()
        # MSE loss (BASELINE)
        if self.mode == MODE_BASE and self.loss_config.recon_weight > 0:
            mse_loss = self._mse_loss(fake_y, y)
            weighted_mse_loss = self.loss_config.recon_weight * mse_loss
            loss += weighted_mse_loss
            if summarize:
                scalar['loss/g/mse'] = mse_loss.detach()
                scalar['loss/g/total'] += weighted_mse_loss.detach()
        # backprop before accumulating mr gradients
        if isinstance(loss, torch.Tensor):
            loss.backward()
        # MR loss (MR)
        if self.mode == MODE_MR:
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
                    image[image_id] = (image[image_id] + 1) / 2
        # Optimize the network
        self.clip_grad(self.optim_g, self.config.g_optimizer.clip_grad)
        self.optim_g.step()
        return {'scalar': scalar, 'histogram': histogram, 'image': image}

    def optimize_p(self, x, y, step, summarize=False):
        # MLE loss (PRED)
        assert self.mode == MODE_PRED
        # MLE loss (PRED)
        mean, log_var = self.net_p(x)
        loss = self.mle_loss(mean, log_var, y)
        # backprop & minimize
        self.optim_p.zero_grad()
        loss.backward()
        self.clip_grad(self.optim_p, self.config.p_optimizer.clip_grad)
        self.optim_p.step()
        return {
            'scalar': {
                'loss/p/mle': loss
            }
        }

    def summarize(self, writer: SummaryWriter, step,
                  train_sample_iterator: DataIterator,
                  val_sample_iterator: DataIterator):
        with torch.no_grad():
            for iter_type, iterator in (
                    ('train', train_sample_iterator),
                    ('val', val_sample_iterator)
            ):
                mle_loss = 0.
                sample_2nds = []
                pred_log_2nds = []
                for i in range(len(iterator)):
                    x, y = next(iterator)
                    h, w = y.size()[2:4]
                    if self.mode == MODE_BASE:
                        self.net_g.eval()
                        num_samples = 12
                        g_x = x.expand(num_samples, -1, -1, -1)
                        samples = self.net_g(g_x)[0]
                        samples_reshaped = samples.view(
                            1, num_samples, *list(samples.size()[1:])
                        )
                        sample_1st, sample_2nd = self.sample_statistics(
                            samples_reshaped
                        )
                        zeros = torch.zeros_like(y[:, :3, ...])
                        self.net_g.train()
                        collage = torch.cat([
                            torch.cat([
                                (F.interpolate(x[:, :3, ...], [h, w]) + 1) / 2,
                                (y + 1) / 2, zeros, zeros, zeros, zeros
                            ], dim=3),
                            torch.cat([
                                (fy.unsqueeze(0) + 1) / 2 for fy in
                                torch.unbind(samples[:num_samples // 2])
                            ], dim=3),
                            torch.cat([
                                (fy.unsqueeze(0) + 1) / 2 for fy in
                                torch.unbind(samples[num_samples // 2:])
                            ], dim=3)
                        ], dim=2)
                        sample_2nds.append(sample_2nd)
                        writer.add_image(
                            'g/{}/{}'.format(iter_type, i), collage, step
                        )
                    elif self.mode == MODE_MR:
                        pred_1st, pred_log_2nd = self.net_p(x)
                        self.net_g.eval()
                        num_samples = 12
                        g_x = x.expand(num_samples, -1, -1, -1)
                        samples = self.net_g(g_x)[0]
                        self.net_g.train()
                        sample_1st, sample_2nd = self.sample_statistics(samples)
                        sample_log_2nd = (sample_2nd + 1e-4).log()
                        log_2nds = torch.cat([pred_log_2nd, sample_log_2nd], 3)
                        uncertainty = self._build_uncertainty_image(log_2nds)
                        collage = [
                            torch.cat([
                                (F.interpolate(x[:, :3, ...], [h, w]) + 1) / 2,
                                (y + 1) / 2,
                                (pred_1st + 1) / 2,
                                (sample_1st + 1) / 2,
                                (uncertainty + 1) / 2,
                            ], dim=3),
                            torch.cat([
                                (fy.unsqueeze(0) + 1) / 2 for fy in
                                torch.unbind(samples[:num_samples // 2])
                            ], dim=3),
                            torch.cat([
                                (fy.unsqueeze(0) + 1) / 2 for fy in
                                torch.unbind(samples[num_samples // 2:])
                            ], dim=3)
                        ]
                        collage = torch.cat(collage, dim=2)
                        writer.add_image(
                            'collage/{}/{}'.format(iter_type, i), collage, step
                        )
                    elif self.mode == MODE_PRED:
                        self.net_p.eval()
                        pred_1st, pred_log_2nd = self.net_p(x)
                        self.net_p.train()
                        uncertainty = self._build_uncertainty_image(
                            pred_log_2nd
                        )
                        collage = torch.cat([
                            torch.cat([
                                (F.interpolate(x[:, :3, ...], [h, w]) + 1) / 2,
                                (y + 1) / 2
                            ], dim=3),
                            torch.cat([
                                (pred_1st + 1) / 2,
                                (uncertainty + 1) / 2,
                            ], dim=3)
                        ], dim=2)
                        pred_log_2nds.append(pred_log_2nd)
                        writer.add_image(
                            'p/{}/{}'.format(iter_type, i), collage, step
                        )

                        # Validation loss
                        if iter_type == 'val':
                            mle_loss += self.mle_loss(pred_1st, pred_log_2nd, y)

                if self.mode == MODE_PRED and iter_type == 'val':
                    pred_log_2nd = torch.stack(pred_log_2nds)
                    pred_2nd = torch.exp(pred_log_2nd)
                    writer.add_scalar(
                        'variance/pred', pred_2nd.mean(), step
                    )
                    writer.add_histogram(
                        'variance_hist/pred', pred_2nd, step
                    )
                if self.mode == MODE_BASE and iter_type == 'val':
                    sample_2nd = torch.stack(sample_2nds)
                    writer.add_scalar(
                        'variance/sample', sample_2nd.mean(), step
                    )
                    writer.add_histogram(
                        'variance_hist/sample', sample_2nd, step
                    )
                if self.mode == MODE_PRED and iter_type == 'val':
                    writer.add_scalar(
                        'loss/p/mle_val', mle_loss / len(iterator), step
                    )

    def build_d_input(self, x, samples):
        return samples

    def mle_target(self, y):
        return y
