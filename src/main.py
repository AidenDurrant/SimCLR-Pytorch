#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import random
import configargparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from train import finetune, evaluate, pretrain, supervised
from datasets import get_dataloaders
from utils import experiment_config, print_network, init_weights
import model.network as models


warnings.filterwarnings("ignore")

default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

parser = configargparse.ArgumentParser(
    description='Pytorch SimCLR', default_config_files=[default_config])
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset, (Options: cifar10, cifar100, stl10, imagenet, tinyimagenet).')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
parser.add_argument('--model', default='resnet18',
                    help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
parser.add_argument('--n_epochs', type=int, default=1000,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--finetune_learning_rate', type=float, default=0.1,
                    help='Starting Learing Rate for Linear Classification Training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='lars',
                    help='Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--finetune_optimiser', default='sgd',
                    help='Finetune Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--patience', default=50, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--temperature', type=float, default=0.5, help='NT_Xent Temperature Factor')
parser.add_argument('--jitter_d', type=float, default=1.0,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')
parser.add_argument('--no_twocrop', dest='twocrop', action='store_false',
                    help='Whether or Not to Use Two Crop Augmentation, Used to Create Two Views of the Input for Contrastive Learning. (Default: True)')
parser.set_defaults(twocrop=True)
parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--no_distributed', dest='distributed', action='store_false',
                    help='Whether or Not to Use Distributed Training. (Default: True)')
parser.set_defaults(distributed=True)
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
parser.set_defaults(finetune=False)
parser.add_argument('--supervised', dest='supervised', action='store_true',
                    help='Perform Supervised Pre-Training. (Default: False)')
parser.set_defaults(supervised=False)


def setup(distributed):
    """ Sets up for optional distributed training.
    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed

    Taken from https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate

    """
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 420
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank


def main():
    """ Main """

    # Arguments
    args = parser.parse_args()

    # Setup Distributed Training
    device, local_rank = setup(distributed=args.distributed)

    # Get Dataloaders for Dataset of choice
    dataloaders, args = get_dataloaders(args)

    # Setup logging, saving models, summaries
    args = experiment_config(parser, args)

    # Get available models from /model/network.py
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # If model exists
    if any(args.model in model_name for model_name in model_names):

        # Load model
        base_encoder = getattr(models, args.model)(
            args, num_classes=args.n_classes)  # Encoder

        proj_head = models.projection_MLP(args)
        sup_head = models.Sup_Head(args)

    else:
        raise NotImplementedError("Model Not Implemented: {}".format(args.model))

    # Remove last FC layer from resnet
    base_encoder.fc = nn.Sequential()

    # Place model onto GPU(s)
    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(6)  # n cpu threads / n processes per node

        base_encoder = DistributedDataParallel(base_encoder.cuda(),
                                               device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        proj_head = DistributedDataParallel(proj_head.cuda(),
                                            device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

        sup_head = DistributedDataParallel(sup_head.cuda(),
                                           device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

        # Only print from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:
        # If non Distributed use DataParallel
        if torch.cuda.device_count() > 1:
            base_encoder = nn.DataParallel(base_encoder)
            proj_head = nn.DataParallel(proj_head)
            sup_head = nn.DataParallel(sup_head)

        print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

        base_encoder.to(device)
        proj_head.to(device)
        sup_head.to(device)

        args.print_progress = True

    # Print Network Structure and Params
    if args.print_progress:
        print_network(base_encoder, args)  # prints out the network architecture etc
        logging.info('\npretrain/train: {} - valid: {} - test: {}'.format(
            len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    # launch model training or inference
    if not args.finetune:

        ''' Pretraining / Finetuning / Evaluate '''

        if not args.supervised:
            # Pretrain the encoder and projection head
            proj_head.apply(init_weights)

            pretrain(base_encoder, proj_head, dataloaders, args)
        else:
            supervised(base_encoder, sup_head, dataloaders, args)

        print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

        # Load the pretrained model
        checkpoint = torch.load(args.load_checkpoint_dir)

        # Load the encoder parameters
        base_encoder.load_state_dict(checkpoint['encoder'])

        # Initalize weights of the supervised / classification head
        sup_head.apply(init_weights)

        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, sup_head, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, sup_head, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()
    else:

        ''' Finetuning / Evaluate '''

        # Do not Pretrain, just finetune and inference

        print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

        # Load the pretrained model
        checkpoint = torch.load(args.load_checkpoint_dir)

        # Load the encoder parameters
        base_encoder.load_state_dict(checkpoint['encoder'])  # .cuda()

        # Initalize weights of the supervised / classification head
        sup_head.apply(init_weights)

        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, sup_head, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, sup_head, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
