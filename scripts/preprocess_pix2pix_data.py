import argparse
import os.path
import torch
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True)
parser.add_argument('--dst', required=True)
parser.add_argument('--size', required=True)
parser.add_argument('--random-flip', action='store_true')
parser.add_argument('--random-rotate', action='store_true')


def main():
    args = parser.parse_args()
    size = int(args.size)
    print('size: {} | random flip: {} | random rotate: {}'
          .format(size, args.random_flip, args.random_rotate))

    for mode in ['train', 'val']:
        src_dir = os.path.join(args.src, mode)
        dst_dir = os.path.join(args.dst, mode)

        items = [
            os.path.join(src_dir, name)
            for name in os.listdir(src_dir)
        ]

        index = 0
        print('Processing %s...' % mode)
        os.makedirs(dst_dir, mode=0o755, exist_ok=True)
        for item in tqdm(items):
            results = []
            raw_image = Image.open(item)

            # Split
            raw_image = F.to_tensor(raw_image)
            image = raw_image[:, :, :raw_image.size(2)//2]
            label = raw_image[:, :, raw_image.size(2)//2:]
            pil_image = F.to_pil_image(image)
            pil_label = F.to_pil_image(label)

            # Resize
            pil_image = F.resize(pil_image, (size, size))
            pil_label = F.resize(pil_label, (size, size))
            results.append((pil_image, pil_label))

            # Rotation (x4)
            if args.random_rotate and mode == 'train':
                for degree in [90, 180, 270]:
                    aug_image = F.rotate(pil_image, degree)
                    aug_label = F.rotate(pil_label, degree)
                    results.append((aug_image, aug_label))

            # Flip and rotate (x4)
            if args.random_flip and mode == 'train':
                pil_image = F.hflip(pil_image)
                pil_label = F.hflip(pil_label)
                results.append((pil_image, pil_label))

                if args.random_rotate:
                    for degree in [90, 180, 270]:
                        aug_image = F.rotate(pil_image, degree)
                        aug_label = F.rotate(pil_label, degree)
                        results.append((aug_image, aug_label))

            # Save results
            for image, label in results:
                merged = torch.cat([
                    F.to_tensor(image), F.to_tensor(label)
                ], 2)
                merged = F.to_pil_image(merged)
                merged.save(os.path.join(dst_dir, '%d.jpg' % index))
                index += 1


if __name__ == '__main__':
    main()
