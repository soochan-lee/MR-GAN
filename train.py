from tensorboardX import SummaryWriter
import yaml
import time
import torch

from models.base import BaseModel
from data import DataIterator, SubsetSequentialSampler, InfiniteRandomSampler
from constants import MODE_PRED
from utils import namedtuple_to_dict


def train(model: BaseModel, config, train_dataset, val_dataset, step=0):
    train_iterator = DataIterator(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.data.num_workers,
        sampler=InfiniteRandomSampler(train_dataset)
    )

    # Prepare for summary
    writer = SummaryWriter(config.log_dir)
    config_str = yaml.dump(namedtuple_to_dict(config))
    writer.add_text('config', config_str)
    train_sampler = SubsetSequentialSampler(
        train_dataset, config.summary.train_samples)
    val_sampler = SubsetSequentialSampler(
        val_dataset, config.summary.val_samples
    )
    train_sample_iterator = DataIterator(
        train_dataset.for_summary(), sampler=train_sampler, num_workers=2)
    val_sample_iterator = DataIterator(
        val_dataset.for_summary(), sampler=val_sampler, num_workers=2)

    # Training loop
    start_time = time.time()
    start_step = step
    while True:
        step += 1
        save_summary = step % config.summary_step == 0
        d_summary, g_summary, p_summary = None, None, None
        if config.mode == MODE_PRED:
            if model.lr_sched_p is not None:
                model.lr_sched_p.step()
            x, y = next(train_iterator)
            p_summary = model.optimize_p(
                x, y, step=step, summarize=save_summary)

        else:
            if model.lr_sched_d is not None:
                model.lr_sched_d.step()

            x, y = next(train_iterator)
            summarize_d = save_summary and config.d_updates_per_step == 1
            d_summary = model.optimize_d(
                x, y, step=step, summarize=summarize_d)
            for i in range(config.d_updates_per_step - 1):
                x, y = next(train_iterator)
                summarize_d = save_summary and (
                        i == config.d_updates_per_step - 2)
                d_summary = model.optimize_d(
                    x, y, step=step, summarize=summarize_d)

            if model.lr_sched_g is not None:
                model.lr_sched_g.step()

            summarize_g = save_summary and config.g_updates_per_step == 1
            g_summary = model.optimize_g(
                x, y, step=step, summarize=summarize_g)
            for i in range(config.g_updates_per_step - 1):
                x, y = next(train_iterator)
                summarize_g = save_summary and (
                        i == config.g_updates_per_step - 2)
                g_summary = model.optimize_g(
                    x, y, step=step, summarize=summarize_g)

        # Print status
        elapsed_time = time.time() - start_time
        elapsed_step = step - start_step
        print(
            '\r[Step %d] %s' % (
                step, time.strftime('%H:%M:%S', time.gmtime(elapsed_time))),
            end='')
        if elapsed_time > elapsed_step:
            print(' | %.2f s/it' % (elapsed_time / elapsed_step), end='')
        else:
            print(' | %.2f it/s' % (elapsed_step / elapsed_time), end='')

        if step % config.ckpt_step == 0:
            model.save(step)

        if save_summary:
            # Save summaries from optimization process
            for summary in [p_summary, d_summary, g_summary]:
                if summary is None:
                    continue
                model.write_summary(writer, summary, step)

            # Summarize learning rates and gradients
            for component, optimizer in [
                ('d', model.optim_d), ('g', model.optim_g),
                ('p', model.optim_p),
            ]:
                if optimizer is None:
                    continue

                for i, group in enumerate(optimizer.param_groups):
                    writer.add_scalar(
                        'lr/%s/%d' % (component, i), group['lr'], step)
                    grads = []
                    for param in group['params']:
                        if param.grad is not None:
                            grads.append(param.grad.data.view([-1]))
                    if grads:
                        grads = torch.cat(grads, 0)
                        writer.add_histogram(
                            'grad/%s/%d' % (component, i), grads, step)

            # Custom summaries
            model.summarize(
                writer, step, train_sample_iterator, val_sample_iterator)
