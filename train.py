import time
import os, sys, yaml
import argparse

import torch
from torch._C import default_generator
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import logging
import numpy as np
from tqdm import tqdm, trange

from src import config, data
from src.checkpoints import CheckpointIO 

if __name__ == '__main__':
    # set random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training ')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of seconds'
                            'with exit code 2.')

    args = parser.parse_args()
    try:
        with open(args.filename, 'r') as file:
            try:
                config_vae = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                config_vae = None
    except:
        config_vae = None

    # config module mainly to load config, dataset, network
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu", args.local_rank)
    print(device, args.local_rank)
    cfg['training']['local_rank'] = args.local_rank

    # set logger
    logger_py = logging.getLogger(__name__)

    if cfg['training']['multi_gpu']:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

        # set logger
        os.makedirs(cfg['training']['out_dir'], exist_ok=True)
        logfile = os.path.join(cfg['training']['out_dir'],
                            cfg['training']['logfile'])
        logger_py.setLevel(level=logging.INFO if dist.get_rank() == 0 else logging.WARNING)

        handler = logging.FileHandler(logfile, mode='a', encoding='UTF-8')
        handler.setLevel(logging.INFO if dist.get_rank() == 0 else logging.WARNING)
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)

        logger_py.addHandler(handler)
    else:
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        config.set_logger(cfg)

    # shorthands
    out_dir = cfg['training']['out_dir']
    backup_every = cfg['training']['backup_every']
    exit_after = args.exit_after
    lr = cfg['training']['learning_rate']
    batch_size = cfg['training']['batch_size'] # number of rays processed in parallel, decrease if running out of memory
    batch_size_val = cfg['training']['batch_size_val']
    n_workers = cfg['training']['n_workers'] # number of workers when loading data
    t0 = time.time()

    # if necessary, select the model according the selection metric and mode
    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                        'either maximize or minimize.')

    # load train and validation dataset
    train_dataset = config.get_dataset("train", cfg)
    train_sampler = None
    if cfg['training']['multi_gpu']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False,
            collate_fn=data.collate_remove_none, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False,
            collate_fn=data.collate_remove_none, )

    val_dataset = config.get_dataset("val", cfg)
    if cfg['training']['multi_gpu']:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size_val, num_workers=n_workers,
            shuffle=False, collate_fn=data.collate_remove_none, sampler=val_sampler)
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=n_workers,
            shuffle=False, collate_fn=data.collate_remove_none,)
    
    viz_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=n_workers, shuffle=False)

    visualize_iter = iter(viz_loader)

    # Initialize training
    model = config.get_model(cfg, device=device, len_dataset=len(train_dataset), config=config_vae)
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr)

    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    # load model parameters from file ( if exists )
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt', device=device)
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    
    metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)
    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print('Current best validation metric (%s): %.8f'
        % (model_selection_metric, metric_val_best))

    # optimizer scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, cfg['training']['scheduler_milestones'],
        gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)

    # load summary writer
    if cfg['training']['multi_gpu'] and dist.get_rank() == 0:
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    elif not cfg['training']['multi_gpu']:
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    else:
        logger = None

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    logger_py.info(model)
    logger_py.info('Total number of parameters: %d' % nparameters)
    t0b = time.time()

    is_postnet_fixed = False
    while True:
        epoch_it += 1
        if cfg['training']['multi_gpu']:
            train_sampler.set_epoch(epoch_it)
            val_sampler.set_epoch(epoch_it)

        for batch, idx in tqdm(train_loader, disable=args.local_rank != 0):
            if isinstance(batch, dict) :
                for key, value in batch.items():
                    batch[key] = value.to(device)
                    
            else:
                batch = batch.to(device)
            it += 1

            if is_postnet_fixed == False and it > 100000:
                if trainer.multi_gpu:
                    for p in trainer.model.module.post_fusion_unet.parameters():
                        p.requires_grad = False
                    trainer.model.module.post_fusion_unet.eval()
                else:
                    for p in trainer.model.post_fusion_unet.parameters():
                        p.requires_grad = False
                    trainer.model.post_fusion_unet.eval()
                is_postnet_fixed = True

            loss, loss_all = trainer.train_step(batch, it=it, seed=0)

            if 'loss_rgb' in loss_all:
                psnr = -10. * np.log(loss_all['loss_rgb']) / np.log(10)
            else:
                psnr = 0

            if logger is not None:
                logger.add_scalar('train/psnr', psnr, it)
                for loss_type in loss_all:
                    logger.add_scalar('train/'+loss_type, loss_all[loss_type], it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                logger_py.info('[Epoch %02d] it=%03d, loss=%.4f, psnr=%.4f, time=%.4f'
                            % (epoch_it, it, loss, psnr, time.time() - t0b))
                for loss_type in loss_all:
                    logger_py.info('%s=%.4f' % (loss_type, loss_all[loss_type]))
                logger_py.info('\n')
                t0b = time.time()

            if visualize_every > 0 and (it % visualize_every) == 0 and args.local_rank == 0:
                try:
                    visualize_value, _ = next(visualize_iter) 
                except StopIteration:
                    visualize_iter = iter(viz_loader)
                    visualize_value, _ = next(visualize_iter) 
                    
                for key, value in visualize_value.items():
                    visualize_value[key] = value.to(device)

                logger_py.info('Visualizing and evaluating one data on tensorboard')
                if cfg['training']['stage'] == 'stage1' or cfg['training']['stage'] == 'stage1_stage2':
                    trainer.visualize(visualize_value, logger, it)
                    
            # Save checkpoint
            if checkpoint_every > 0 and (it % checkpoint_every) == 0 and args.local_rank == 0:
                logger_py.info('Saving checkpoint')
                # print('Saving checkpoint')
                if cfg['training']['multi_gpu']:
                    if dist.get_rank() == 0:
                        checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                        loss_val_best=metric_val_best)
                else:
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                logger_py.info('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0 and it != 0 and args.local_rank == 0:
                logger_py.info('Doing validation!')
                eval_dict = trainer.evaluate(val_loader, 
                                             focal_length=cfg['data']['focal_length'], 
                                             batch_size=batch_size, it=it)
                metric_val = eval_dict[model_selection_metric]
                logger_py.info('Validation metric (%s): %.4f'
                            % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    if logger is not None:
                        logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger_py.info('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.backup_model_best('model_best.pt')
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

            # Exit if necessary
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                logger_py.info('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
                exit(3)

        # Make scheduler step after full epoch
        scheduler.step()
    
