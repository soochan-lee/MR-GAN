import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from constants import MODE_BASE, MODE_PRED, MODE_MR
from data import DataIterator
from .base import BaseModel, generate_noise


class Generator(nn.Module):
    def __init__(self, config, as_predictor=False):
        super().__init__()
        self.config = config
        self.as_predictor = as_predictor
        self.noise_dim = nd = config.model.noise_dim
        self.noise_type = config.model.noise_type

        # heads
        self.mean_conv = nn.Conv2d(
            32 + nd[0] if nd[0] > 0 and not as_predictor else 32, 3,
            3, 1, 1
        )
        self.dispersion_conv = \
            nn.Conv2d(32, 3, 3, 1, 1) if as_predictor else None

        # hidden layers
        self.layers = nn.ModuleList([
            # conv1
            nn.Sequential(
                nn.Conv2d(4, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            # conv2
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # conv3
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # conv4
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # conv5
            nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # conv6
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[11] if nd[11] > 0 and not as_predictor else 256,
                    256, 3, 1, 1
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # dil1
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[10] if nd[10] > 0 and not as_predictor else 256,
                    256, 3, 1, 2, dilation=2
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # dil2
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[9] if nd[9] > 0 and not as_predictor else 256,
                    256, 3, 1, 4, dilation=4
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # dil3
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[8] if nd[8] > 0 and not as_predictor else 256,
                    256, 3, 1, 8, dilation=8
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # dil4
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[7] if nd[7] > 0 and not as_predictor else 256,
                    256, 3, 1, 16, dilation=16
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # conv7
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[6] if nd[6] > 0 and not as_predictor else 256,
                    256, 3, 1, 1
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # conv8
            nn.Sequential(
                nn.Conv2d(
                    256 + nd[5] if nd[5] > 0 and not as_predictor else 256,
                    256, 3, 1, 1
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # deconv1
            nn.Sequential(
                nn.ConvTranspose2d(
                    256 + nd[4] if nd[4] > 0 and not as_predictor else 256,
                    128, 4, 2, 1
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # conv9
            nn.Sequential(
                nn.Conv2d(
                    128 + nd[3] if nd[3] > 0 and not as_predictor else 128,
                    128, 3, 1, 1
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # deconv2
            nn.Sequential(
                nn.ConvTranspose2d(
                    128 + nd[2] if nd[2] > 0 and not as_predictor else 128,
                    64, 4, 2, 1
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            # conv10
            nn.Sequential(
                nn.Conv2d(
                    64 + nd[1] if nd[1] > 0 and not as_predictor else 64,
                    32, 3, 1, 1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, x, num_samples=1):
        if num_samples > 1:
            x = x\
                .unsqueeze(1)\
                .expand(-1, num_samples, -1, -1, -1)\
                .contiguous()\
                .view(x.size(0) * num_samples, x.size(1), x.size(2), x.size(3))

        images = x[:, :3, ...]
        masks = x[:, -1:, ...]
        net = x
        # run through the hidden layers
        for i, layer in enumerate(self.layers):
            reversed_index = len(self.layers) - i - 1
            net = layer(net)
            if not self.as_predictor and (
                    len(self.noise_dim) > reversed_index and
                    self.noise_dim[reversed_index] > 0
            ):
                noise = generate_noise(
                    self.noise_type,
                    self.noise_dim[reversed_index],
                    like=net
                )
                net = torch.cat([net, noise], 1)

        mean = F.sigmoid(self.mean_conv(net))
        mean = mean * masks + images * (1. - masks)

        if self.as_predictor:
            dispersion = self.dispersion_conv(net)
            dispersion = (
                dispersion * masks +
                self.config.log_dispersion_min * (1. - masks)
            )
            return mean, dispersion
        else:
            return mean, None


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.global_d = GlobalDiscriminator(config)
        self.local_d = LocalDiscriminator(config)
        self.linear = nn.Linear(1024 * 2, 1)

    def forward(self, x_y):
        x, y = x_y
        local_boxes = x.local_boxes
        global_output = self.global_d(y)
        local_output = self.local_d(self._local(y, local_boxes))
        return self.linear(torch.cat([global_output, local_output], 1))

    def _local(self, images, local_boxes):
        local_images = []
        for image, local_box in zip(images, local_boxes):
            y1, x1, y2, x2 = local_box
            local_images.append(image[:, y1:y2, x1:x2])
        return torch.stack(local_images)


class GlobalDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(
            512 * (config.data.random_crop // (2 ** 6)) ** 2,
            1024
        )
        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # conv3
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(512, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv6
            nn.Conv2d(512, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, y):
        net = self.layers(y)
        net = net.view(net.size(0), -1)
        return self.linear(net)


class LocalDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(
            512 * (config.data.local_size // (2 ** 5)) ** 2,
            1024
        )
        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # conv3
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(512, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, y):
        net = self.layers(y)
        net = net.view(net.size(0), -1)
        return self.linear(net)


class GLCIC(BaseModel):
    name = 'glcic'

    def __init__(self, config):
        self.t_g = config.g_pretrain_step
        self.t_d = config.d_pretrain_step
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
        torch.cuda.empty_cache()
        image = {}
        assert self.mode in (MODE_BASE, MODE_MR)
        if step < self.t_g or self.loss_config.gan_weight <= 0:
            return {}
        # gan loss (BASELINE || MR)
        fake_y = self.net_g(x)[0]
        fake_v = self.net_d((x, fake_y))

        # jitter real y
        if self.config.data.real_jitter > 0:
            mask = x[:, -1:, ...]
            jitter = torch.randn(y.size(0), y.size(1), 1, 1, device='cuda')
            real_y = y + mask * self.config.data.real_jitter * jitter
            real_y = torch.clamp(real_y, 0., 1.)
        else:
            real_y = y

        real_v = self.net_d((x, real_y))
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
                image[real_image_id] = real_y[i]
                image[fake_image_id] = fake_y[i]
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
        torch.cuda.empty_cache()
        assert self.mode in (MODE_BASE, MODE_MR)
        # prepare some accumulators
        scalar = {'loss/g/total': 0.}
        histogram = {}
        image = {}
        loss = 0.
        self.optim_g.zero_grad()
        # GAN loss (BASELINE)
        fake_y = self.net_g(x)[0]
        if self.mode == MODE_BASE and (step > self.t_g + self.t_d):
            fake_v = self.net_d((x, fake_y))
            gan_loss = self.gan_loss(fake_v, True)
            weighted_gan_loss = self.loss_config.gan_weight * gan_loss
            loss += weighted_gan_loss
            if summarize:
                scalar['loss/g/gan'] = gan_loss.detach()
                scalar['loss/g/total'] += weighted_gan_loss.detach()
        # MSE loss (BASELINE)
        if self.mode == MODE_BASE and self.loss_config.recon_weight > 0 and (
                step > self.t_g + self.t_d or
                step < self.t_g
        ):
            mse_loss = self._mse_loss(fake_y, y)
            weighted_mse_loss = self.loss_config.recon_weight * mse_loss
            loss += weighted_mse_loss
            if summarize:
                scalar['loss/g/mse'] = mse_loss.detach()
                scalar['loss/g/total'] += weighted_mse_loss.detach()
        # backprop before accumulating mm gradients
        if isinstance(loss, torch.Tensor):
            loss.backward()

        # MR loss (MR)
        if self.mode == MODE_MR:
            mm_summaries = self.accumulate_mm_grad(x, y, summarize)
            mm_scalar = mm_summaries['scalar']
            mm_histogram = mm_summaries['histogram']
            mm_image = mm_summaries['image']
            torch.cuda.empty_cache()
            if summarize:
                scalar['loss/g/total'] += mm_scalar['loss/g/total']
                del mm_scalar['loss/g/total']
                scalar.update(mm_scalar)
                histogram.update(mm_histogram)
                for i in range(min(16, self.config.num_mm)):
                    image_id = 'train_samples/%d' % i
                    if image_id in image:
                        image[image_id] = torch.cat([
                            image[image_id], mm_image[image_id]
                        ], 2)
                    else:
                        image[image_id] = mm_image[image_id]
                    image[image_id] = image[image_id]
        # Optimize the network
        self.clip_grad(self.optim_g, self.config.g_optimizer.clip_grad)
        self.optim_g.step()
        return {'scalar': scalar, 'histogram': histogram, 'image': image}

    def optimize_p(self, x, y, step, summarize=False):
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
                        zeros = torch.zeros_like(x[:, :3, ...])
                        self.net_g.train()
                        collage = torch.cat([
                            torch.cat(
                                [x[:, :3, ...], y, zeros, zeros, zeros, zeros],
                                dim=3
                            ),
                            torch.cat([
                                fy.unsqueeze(0) for fy in
                                torch.unbind(samples[:num_samples // 2])
                            ], dim=3),
                            torch.cat([
                                fy.unsqueeze(0) for fy in
                                torch.unbind(samples[num_samples // 2:])
                            ], dim=3)
                        ], dim=2)
                        sample_2nds.append(sample_2nd)
                        writer.add_image(
                            'g/{}/{}'.format(iter_type, i), collage, step
                        )
                    elif self.mode == MODE_MR:
                        mask = x[:, -1:, ...]
                        pred_1st, pred_log_2nd = self.net_p(x)
                        self.net_g.eval()
                        num_samples = 12
                        g_x = x.expand(num_samples, -1, -1, -1)
                        samples = self.net_g(g_x)[0]
                        self.net_g.train()
                        sample_1st, sample_2nd = self.sample_statistics(samples)
                        sample_log_2nd = (sample_2nd + 1e-4).log()
                        sample_log_2nd = (
                            sample_log_2nd * mask +
                            self.config.log_dispersion_min * (1. - mask)
                        )
                        log_2nds = torch.cat([pred_log_2nd, sample_log_2nd], 3)
                        uncertainty = self._build_uncertainty_image(log_2nds)
                        uncertainty = (uncertainty + 1.) * 0.5
                        collage = [
                            torch.cat([
                                x[:, :3, ...], y, pred_1st,
                                sample_1st, uncertainty,
                            ], dim=3),
                            torch.cat([
                                fy.unsqueeze(0) for fy in
                                torch.unbind(samples[:num_samples // 2])
                            ], dim=3),
                            torch.cat([
                                fy.unsqueeze(0) for fy in
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
                        uncertainty = (uncertainty + 1.) * 0.5
                        collage = torch.cat([
                            torch.cat([x[:, :3, ...], y], dim=3),
                            torch.cat([pred_1st, uncertainty], dim=3)
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

    def mle_target(self, y):
        return y

    def build_d_input(self, x, samples):
        num_mm = self.config.num_mm
        num_samples = self.config.num_mm_samples
        local_box_dup = x.local_boxes[:num_mm].unsqueeze(1)
        local_box_dup = local_box_dup.expand(-1, num_samples, -1)
        local_box_dup = local_box_dup.contiguous().view(
            num_mm * num_samples, local_box_dup.size(2)
        )

        x_dup = x[:num_mm].unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
        x_dup = x_dup.contiguous().view(
            num_mm * num_samples, *list(x_dup.size()[2:]))

        x_dup.local_boxes = local_box_dup

        return x_dup, samples
