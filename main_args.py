import argparse
import inspect

import utils

# Load loss, loader and transforms
from model_builders.multi_head import MultiHeadClassifier
from utils import kv_pair


def get_args_parser():
    parser = argparse.ArgumentParser('MI clustering', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='clip_ViT-B/32', type=str,
                        help="""Name of architecture to train. For quick experiments with ViTs,
                        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--datapath', default='./data', type=str)
    parser.add_argument('--embed_dim', default=None, type=int)
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update.
        The value is increased to max_momentum_teacher during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--max_momentum_teacher', default=1.0, type=float)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--head_dropout_prob', default=0.0, type=float,
                        help="Dropout probability in projection head (Default: 0.0)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.1, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.1, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.0001, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")

    bs_group = parser.add_mutually_exclusive_group()
    bs_group.add_argument('--batch_size_per_gpu', default=64, type=int,
                          help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    bs_group.add_argument('--batch_size', default=None, type=int,
                          help='Total batch size')

    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop and aug
    parser.add_argument('--vit_image_size', type=int, default=224, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--image_size', type=int, default=32, help="""image size of in-distibution data. 
        negative samples are first resized to image_size and then inflated to vit_image_size. This
        ensures that aux samples have same resolution as in-dist samples""")
    parser.add_argument('--aug_image_size',type=int, default=None,
                       help='Image size for data augmentation. If None, use vit_image_size')
    from augs.augs import IMAGE_AUGMENTATIONS, EMBED_AUGMENTATIONS
    parser.add_argument('--image_aug', choices=IMAGE_AUGMENTATIONS.keys(), default='randaug',
                        help='Augmentation for images')
    parser.add_argument('--embed_aug', choices=EMBED_AUGMENTATIONS.keys(), default='none',
                        help='Augmentations for precomputed embeddings. Only used when --precomputed flag is given')
    parser.add_argument('--aug_args', type=kv_pair, nargs='*', default={})
    parser.add_argument('--num_augs', type=int, default=1)

    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--knn_path', default=None, type=str)
    parser.add_argument('--dataset', default='CIFAR100',
                        choices=['CIFAR100', 'CIFAR10', "STL10", "CIFAR20", "IN1K", "IN50", 'IN100', "IN200"],
                        type=str)
    parser.add_argument('--knn', type=int, default=50, help='Number of nearest neighbors to use')

    parser.add_argument('--output_dir', default=None, type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--out_dim', default=1000, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

    parser.add_argument('--nlayers', default=2, type=int, help='Head layers')
    parser.add_argument('--hidden_dim', default=512, type=int, help="Head's hidden dim")
    parser.add_argument('--bottleneck_dim', default=256, type=int, help="Head's bottleneck dim")
    parser.add_argument('--l2_norm', default=False, action='store_true', help="Whether to apply L2 norm after backbone")
    parser.add_argument('--embed_norm', default=False, action='store_true',
                        help="Whether to normalize embeddings using precomputed mean and std")

    add_from_signature(parser, MultiHeadClassifier, "MultiHeadClassifier")

    parser.add_argument('--disable_ddp', default=False, action='store_true', help="Don't use DDP")

    precomputed_group = parser.add_mutually_exclusive_group()
    precomputed_group.add_argument('--train_backbone', default=False, action='store_true',
                        help="Don't share backbones between teacher and student")
    precomputed_group.add_argument('--precomputed', action='store_true', help="Use precomputed embeddings", default=False)

    parser.add_argument("--loss", default="TEMI", help="The name of one of the classes in losses",
                        choices=['WMI', 'PMI', 'TEMI', 'SCAN'])
    parser.add_argument("--loss-args", type=kv_pair, nargs="*", default={},
                        help="Extra arguments for the loss class")
    parser.add_argument("--loader", default='EmbedNN', help="The name of one of the classes in loaders")
    parser.add_argument("--loader-args", type=kv_pair, nargs="*", default={},
                        help="Extra arguments for the loader class")
    parser.add_argument('--new-run', action='store_true', help="Create a new directory for this run", default=False)

    return parser


def add_from_signature(parser, function, name=None):
    """Add arguments from a function signature to an existing parser."""
    if name is None:
        name = function.__name__
    group = parser.add_argument_group(name)
    signature = inspect.signature(function)
    for name, param in signature.parameters.items():
        default = param.default
        if param.kind == param.VAR_KEYWORD or default is param.empty:
            continue
        try:
            group.add_argument("--" + name, default=param.default, type=type(default))
        except argparse.ArgumentError:
            # Ignore arguments that are already added
            pass


def set_default_args(config):
    if hasattr(config, '__dict__'):
        config = vars(config)
    args = get_args_parser().parse_args([])
    args.__dict__.update(config)
    return args


def process_args(args):
    args.loss_args = dict(args.loss_args)
    args.loader_args = dict(args.loader_args)
    args.aug_args = dict(args.aug_args)
    if not args.knn_path:
        import gen_embeds
        args.knn_path = gen_embeds.get_outpath(args.arch, args.dataset) / 'knn.pt'
    if args.batch_size is not None:
        args.batch_size_per_gpu = None
    return args
