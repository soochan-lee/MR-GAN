#!/usr/bin/env python3
import argparse
import os.path
import torch
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import random


parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True)
parser.add_argument('--dst', required=True)
parser.add_argument('--size', required=True)
parser.add_argument('--val-ratio', default=0.1)


def main():
    args = parser.parse_args()
    size = int(args.size)
    print('size: {}'.format(size,))

    src_dir = args.src
    dst_dir = args.dst

    items = [
        os.path.join(src_dir, name)
        for name in os.listdir(src_dir)
    ]

    train_index = 0
    val_index = 0
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'val')
    os.makedirs(train_dir, mode=0o755, exist_ok=False)
    os.makedirs(val_dir, mode=0o755, exist_ok=False)

    for item in tqdm(items):
        raw_image = Image.open(item)

        # Crop
        raw_image = F.to_tensor(raw_image)
        h = raw_image.size(1)
        w = raw_image.size(2)
        y_offset = (h - w) // 2
        square = raw_image[:, y_offset:y_offset + w, :]
        pil_square = F.to_pil_image(square)

        # Resize
        pil_hi = F.resize(pil_square, (size, size))
        pil_lo = F.resize(pil_hi, (size // 4, size // 4))

        # Pad & concatenate
        hi = F.to_tensor(pil_hi)
        lo = F.to_tensor(pil_lo)
        pad_size = hi.size(1) - lo.size(1)
        pad = torch.zeros(lo.size(0), pad_size, lo.size(2))
        lo_padded = torch.cat([lo, pad], 1)
        result = torch.cat([hi, lo_padded], 2)

        # Save results
        result = F.to_pil_image(result)
        if random.random() > args.val_ratio:
            save_path = os.path.join(train_dir, '%06d.png' % train_index)
            train_index += 1
        else:
            save_path = os.path.join(val_dir, '%06d.png' % val_index)
            val_index += 1

        result.save(save_path)


if __name__ == '__main__':
    main()
