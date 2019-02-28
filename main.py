import argparse
import yaml
import os
from data import DATASETS
from models import MODELS
from train import train
from utils import dict_to_namedtuple
from constants import MODES, MODE_BASE, MODE_MR

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir')
parser.add_argument('--config')
parser.add_argument('--mode', choices=MODES, default=MODE_BASE)
parser.add_argument('--pred-ckpt')
parser.add_argument('--resume-ckpt')
parser.add_argument('--options', '-o', default='')

if __name__ == '__main__':
    args = parser.parse_args()

    assert args.mode in MODES, 'Unknown mode %s' % args.mode
    if args.mode == MODE_MR:
        if not (args.pred_ckpt or args.resume_ckpt):
            print('WARNING: Proxy MR-GAN requires '
                  'checkpoint path of a predictor')

    # Load config
    config_path = args.config
    if args.resume_ckpt and not args.config:
        base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        config_path = os.path.join(base_dir, 'config.yaml')
    config = yaml.load(open(config_path))

    # Override options
    config['mode'] = args.mode
    for option in args.options.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))

        here[keys[-1]] = yaml.load(value)

    # Set log directory
    config['log_dir'] = args.log_dir
    if not args.resume_ckpt and args.log_dir and os.path.exists(args.log_dir):
        print('WARNING: %s already exists' % args.log_dir)
        input('Press enter to continue')

    if args.resume_ckpt and not args.log_dir:
        config['log_dir'] = os.path.dirname(
            os.path.dirname(args.resume_ckpt)
        )

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    if not args.resume_ckpt or args.config:
        config_save_path = os.path.join(config['log_dir'], 'config.yaml')
        yaml.dump(config, open(config_save_path, 'w'))
        print('Config file saved to {}'.format(config_save_path))

    config = dict_to_namedtuple(config)

    # Instantiate dataset
    dataset_factory = DATASETS[config.data.name]
    train_dataset, val_dataset = dataset_factory(config)

    model = MODELS[config.model.name](config)
    model.cuda()

    if args.resume_ckpt:
        print('Resuming checkpoint %s' % args.resume_ckpt)
        step = model.load(args.resume_ckpt)
    else:
        step = 0
    if args.pred_ckpt:
        print('Loading predictor from %s' % args.pred_ckpt)
        model.load_module(model.net_p, args.pred_ckpt)

    train(model, config, train_dataset, val_dataset, step)
