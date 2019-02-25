import copy
import os.path
import numpy as np
import random
import torch
from torchvision.transforms import functional as F
from torch.utils import data
from PIL import Image
from collections import Iterator, Iterable


class InfiniteRandomIterator(Iterator):
    def __init__(self, data_source):
        self.data_source = data_source
        self._perms = [
            torch.randperm(len(self.data_source)).tolist()
            for _ in range(30)
        ]
        self.iterator = iter(random.choice(self._perms))

    def __next__(self):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(random.choice(self._perms))
            idx = next(self.iterator)

        return idx


class InfiniteRandomSampler(data.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return InfiniteRandomIterator(self.data_source)

    def __len__(self):
        return len(self.data_source)


class InfiniteSubsetIterator(Iterator):
    def __init__(self, indices):
        self.indices = indices
        self.iterator = iter(self.indices)

    def __next__(self):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.indices)
            idx = next(self.iterator)

        return idx

    def __len__(self):
        return len(self.indices)


class SubsetSequentialSampler(data.sampler.Sampler):
    def __init__(self, data_source, indices):
        super().__init__(data_source)
        if isinstance(indices, Iterable):
            self.indices = indices
        else:
            self.indices = np.random.choice(
                len(data_source),
                size=indices,
                replace=False
            )

    def __iter__(self):
        return InfiniteSubsetIterator(self.indices)

    def __len__(self):
        return len(self.indices)


class DataIterator(Iterator):
    def __init__(self, dataset: data.Dataset, **kwargs):
        self.data_loader = data.DataLoader(dataset, **kwargs)
        self.epoch = 0
        self.iterator = iter(self.data_loader)

    def __next__(self):
        batch = next(self.iterator)
        batch = self.data_loader.dataset.vector_preprocess(*batch)

        return batch

    def __len__(self):
        return len(self.data_loader)


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def vector_preprocess(self, x, y):
        return x, y

    def for_summary(self):
        return self


# =======================
# Dataset Implementations
# =======================

class Maps(BaseDataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        assert mode in ['train', 'val']
        self.config = config
        self.mode = mode
        self.resize = config.data.resize
        self.random_flip = config.data.random_flip
        self.random_rotate = config.data.random_rotate
        self.resize_shape = [config.data.height, config.data.width * 2]
        if config.data.norm:
            self.mean = torch.tensor(
                config.data.norm.mean, device='cuda'
            ).view(1, 3, 1, 1)
            self.std = torch.tensor(
                config.data.norm.std, device='cuda'
            ).view(1, 3, 1, 1)
        else:
            self.mean = None
            self.std = None

        # Load image paths
        image_dir = os.path.join(config.data.root, mode)
        self.items = [
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
        ]
        self.items = sorted(self.items)

    def __getitem__(self, index):
        raw_image = F.to_tensor(Image.open(self.items[index]))
        if self.resize:
            raw_image = F.resize(raw_image, self.resize_shape)

        w = self.config.data.width
        image = raw_image[:, :, :w]
        label = raw_image[:, :, w:]

        th = self.config.data.train_height
        tw = self.config.data.train_width
        if self.mode == 'train':
            max_offset_y = self.config.data.height - th
            max_offset_x = self.config.data.width - tw

            offset_y = torch.randint(
                low=0, high=max_offset_y, size=[],
                dtype=torch.long
            )
            offset_x = torch.randint(
                low=0, high=max_offset_x, size=[],
                dtype=torch.long
            )
            image = image[:, offset_y:offset_y + th, offset_x:offset_x + tw]
            label = label[:, offset_y:offset_y + th, offset_x:offset_x + tw]

            # Data augmentation
            if self.random_flip or self.random_rotate:
                image = F.to_pil_image(image)
                label = F.to_pil_image(label)

                # Random flip
                if self.random_flip and random.random() < 0.5:
                    image = F.vflip(image)
                    label = F.vflip(label)

                # Random rotation
                if self.random_rotate:
                    degree = random.choice([0, 90, 180, 270])
                    if degree != 0:
                        image = F.rotate(image, degree)
                        label = F.rotate(label, degree)

                image = F.to_tensor(image)
                label = F.to_tensor(label)
        else:
            image = image[:, :th, :tw]
            label = label[:, :th, :tw]

        return label, image

    def __len__(self):
        return len(self.items)

    def vector_preprocess(self, x, y):
        if self.mean is not None:
            x = x.cuda().sub_(self.mean).div_(self.std)
        else:
            x = x.cuda().mul_(2.).sub_(1.)
        y = y.cuda().mul_(2.).sub_(1.)
        return x, y


class Cityscapes(BaseDataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        assert mode in ['train', 'val']
        self.config = config
        self.mode = mode
        if config.data.norm:
            self.mean = torch.tensor(
                config.data.norm.mean, device='cuda'
            ).view(1, 3, 1, 1)
            self.std = torch.tensor(
                config.data.norm.std, device='cuda'
            ).view(1, 3, 1, 1)
        else:
            self.mean = None
            self.std = None

        # Load image paths
        image_dir = os.path.join(config.data.root, mode)
        self.items = sorted([
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
        ])

        data_offset = config.data.data_offset or 0
        data_size = config.data.data_size

        if data_size is not None:
            self.items = self.items[data_offset: data_offset + data_size]
        else:
            self.items = self.items[data_offset:]

    def __getitem__(self, index):
        raw_image = F.to_tensor(Image.open(self.items[index]))
        w = self.config.data.width
        image = raw_image[:, :, :w]
        label = raw_image[:, :, w:]
        return label, image

    def __len__(self):
        return len(self.items)

    def vector_preprocess(self, x, y):
        if self.mean is not None:
            x = x.cuda().sub_(self.mean).div_(self.std)
        else:
            x = x.cuda().mul_(2.).sub_(1.)
        y = y.cuda().mul_(2.).sub_(1.)
        return x, y


class Edges2Shoes(Cityscapes):
    def __getitem__(self, index):
        raw_image = F.to_tensor(Image.open(self.items[index]))
        w = self.config.data.width
        label = raw_image[:, :, :w]
        image = raw_image[:, :, w:]
        return label, image


class CelebA(BaseDataset):
    mean_rgb = [130 / 255, 108 / 255,  96 / 255]

    def __init__(self, config, mode='train'):
        super().__init__()
        assert mode in ['train', 'val']
        self.config = config
        self.mode = mode
        self.mean = None
        self.std = None
        self.size = getattr(config.data, 'size', None)
        self.local_size = getattr(config.data, 'local_size', None)
        self.mask_size = getattr(config.data, 'mask_size', None)
        self.summary = False

        # Load image paths
        image_dir = os.path.join(config.data.root, mode)
        self.items = [
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
        ]
        self.items = sorted(self.items)

    def __getitem__(self, index):
        raw_image = F.to_tensor(Image.open(self.items[index]))
        h = raw_image.size(1)
        w = raw_image.size(2)
        image = raw_image[:, :, :h]
        label = raw_image[:, :w - h, h:]

        return label, image

    def __len__(self):
        return len(self.items)

    def for_summary(self):
        clone = copy.deepcopy(self)
        clone.summary = True
        return clone

    def vector_preprocess(self, x, y):
        if self.config.model.name == 'glcic':
            y = y.cuda()
            x = y.clone()
            local_boxes = []
            masks = []
            for i in range(len(x)):
                if self.summary:
                    seed = x[i]
                else:
                    seed = None
                mask, mask_box, local_box = self._random_mask_in_local_box(seed)
                masks.append(mask)
                local_boxes.append(local_box)
                q1, p1, q2, p2 = mask_box
                x[i, 0, q1:q2, p1:p2] = self.mean_rgb[0]
                x[i, 1, q1:q2, p1:p2] = self.mean_rgb[1]
                x[i, 2, q1:q2, p1:p2] = self.mean_rgb[2]
            # annotate the image tensor
            local_boxes = torch.from_numpy(np.stack(local_boxes)).cuda()
            masks = torch.from_numpy(np.stack(masks)).float().cuda()
            x = torch.cat([x, masks], dim=1)
            x.local_boxes = local_boxes
            return x, y

        else:
            x = x.cuda().mul_(2.).sub_(1.)
            y = y.cuda().mul_(2.).sub_(1.)
            return x, y

    def _random_mask_in_local_box(self, seed=None):
        input_size = self.size
        local_size = self.local_size
        mh_range, mw_range = self.mask_size
        if seed is not None:
            np.random.seed(int(seed.sum() * 100))

        # generate a random mask inside a local box
        max_offset = input_size - local_size
        y1, x1 = np.random.randint(0, max_offset + 1, 2)
        y2, x2 = np.array([y1, x1]) + local_size
        h = np.random.randint(mh_range[0], mh_range[1] + 1)
        w = np.random.randint(mw_range[0], mw_range[1] + 1)
        q1 = y1 + np.random.randint(0, local_size - h + 1)
        p1 = x1 + np.random.randint(0, local_size - w + 1)
        q2 = q1 + h
        p2 = p1 + w
        mask = np.zeros([1, input_size, input_size], dtype=np.float32)
        mask[:, q1:q2, p1:p2] = 1.0
        return mask, np.array([q1, p1, q2, p2]), np.array([y1, x1, y2, x2])


# =================
# Dataset Factories
# =================


def CITYSCAPES(config):
    return Cityscapes(config, 'train'), Cityscapes(config, 'val')


def MAPS(config):
    return Maps(config, 'train'), Maps(config, 'val')


def CELEBA(config):
    return CelebA(config, 'train'), CelebA(config, 'val')


DATASETS = {
    'cityscapes': CITYSCAPES,
    'maps': MAPS,
    'celeba': CELEBA,
}
