
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os

from pathlib import Path

from timm.models import create_model
from timm.utils import ModelEma
from rerank_modeling import vggt_rerank, vggt_classifier, vggt_rerank3t1

from armbench.rerank_datasets import create_rerank_dataset, create_rerank_dataset3t1
from armbench.classifier_datasets import create_classifier_dataset
from beit3_tools.optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit

from armbench.engine_for_rerank import train_one_epoch, get_handler
from beit3_tools.utils import NativeScalerWithGradNormCount as NativeScaler
from beit3_tools import utils
import wandb
import datasets
import random
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser('vggt fine-tuning for reranking', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vggt', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True,
                        choices=['rerank', 'classifier', 'rerank3t1'],
                        help='Name of task to fine-tuning')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--checkpoint_activations', action='store_true', default=None,
                        help='Enable checkpointing to save your memory.')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Finetuning params
    parser.add_argument('--vggt_path', default='',
                        help="vggt's checkpoint path")

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    # distributed parameters

    # wandb options
    parser.add_argument('--enable_wandb', action='store_true', default=False, )

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        # try:
        #     import deepspeed
        #     from deepspeed import DeepSpeedConfig
        #     parser = deepspeed.add_config_arguments(parser)
        #     ds_init = deepspeed.initialize
        # except:
        #     print("Please 'pip install deepspeed==0.4.0'")
        #     exit(0)
        raise Exception("Windows does not support DeepSpeed")
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # datasets.config.IN_MEMORY_MAX_SIZE = 16000000000
    print('Set dataset loading memory size to:', datasets.config.IN_MEMORY_MAX_SIZE)

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.task == "rerank":
        data_loader_train = create_rerank_dataset(args)
        #  set output dir
        exp_tag = f"{args.model}_{args.epochs}epochs_{args.task}_tasks_{args.vggt_path.split('/')[-1]}/"
    elif args.task == 'rerank3t1':
        data_loader_train = create_rerank_dataset3t1(args)
        #  set output dir
        exp_tag = f"{args.model}_{args.epochs}epochs_{args.task}_tasks_{args.vggt_path.split('/')[-1]}/"
    elif args.task == "classifier":
        data_loader_train = create_classifier_dataset(args)
        #  set output dir
        exp_tag = f"{args.model}_{args.epochs}epochs_{args.task}_tasks"


    args.output_dir = os.path.join(args.output_dir, exp_tag)

    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.model.endswith(args.task):
        if args.task in ['rerank', 'rerank3t1', 'classifier']:
            model_config = "%s_%s" % (args.model, args.task)
        else:
            raise Exception("Unknown task %s" % args.task)
    else:
        model_config = args.model

    # initial wandb
    if utils.is_main_process() and args.enable_wandb:
        wandb_config = {
            'lr': args.lr,
            'min_lr': args.min_lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'layer_decay': args.layer_decay,
            'vggt_path': args.vggt_path,
            'task': args.task,
        }

    # print("model_config = %s" % model_config)
    model = create_model(
        model_config,
        pretrained=False,
    )

    if args.vggt_path:
        missing = utils.load_model(args.vggt_path, model)

        for param in model.parameters():
            param.requires_grad = False
    
        for name, param in model.vggt.named_parameters():
            if name in missing:  
                param.requires_grad = True

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)


    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train.dataset))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)


    #num_layers = model_without_ddp.get_num_layers()
    #if args.layer_decay < 1.0:
    #    lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
    #    assigner = LayerDecayValueAssigner(lrs)
    #elif args.task_head_lr_weight > 1:
    #    assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
    #else:
    #    assigner = None

    #if assigner is not None:
    #    print("Assigned values = %s" % str(assigner.values))


    #skip_weight_decay_list = model.no_weight_decay()


    if args.distributed:
        torch.distributed.barrier()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            print("Using torch.nn.parallel.DistributedDataParallel ... on GPU %d" % args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp,
            get_num_layer=None,
            get_layer_scale=None)
        loss_scaler = NativeScaler()


    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    task_handler = get_handler(args)

    # mixup for imagenet
    mixup_fn = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, task_handler, epoch,
            epoch * num_training_steps_per_epoch, lr_schedule_values, loss_scaler,
            args.clip_grad, args.update_freq, model_ema, log_writer, args.task, mixup_fn,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process() and args.enable_wandb:
        wandb.finish()


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
