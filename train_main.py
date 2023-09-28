import os
import sys
import datetime
import time

import math
import json
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import utils
from augs.augs import IMAGE_AUGMENTATIONS, EMBED_AUGMENTATIONS, AugWrapper
import loaders

from torchvision import models as torchvision_models

import losses
from main_args import get_args_parser, process_args
from model_builders import load_model


class TeacherStudentCombo(nn.Module):

    def __init__(self, student, teacher, args):
        super().__init__()
        # synchronize batch norms (if any)
        if utils.has_batchnorms(student) and not args.disable_ddp:
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # teacher and student start with the same weights
        teacher.load_state_dict(student.state_dict())
        # Hacky
        if not args.train_backbone:
            student.backbone = teacher.backbone
        elif not args.req_grad:
            print('WARNING: args.train_backbone=True, but args.req_grad=False. '
                  'This is probably not what you want.')
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {args.arch} network.")

        self.args = args
        self.student = student
        self.teacher = teacher

    def forward(self, images):
        if self.args.train_backbone:
            return self.teacher(images), self.student(images)
        embed = self.teacher.backbone_embed(images)
        return self.teacher.apply_head(embed), self.student.apply_head(embed)

    @property
    def module(self):
        return self

    def student_dict(self):
        if self.args.train_backbone:
            return self.student.state_dict()
        return OrderedDict([(k, v) for k, v in self.student.state_dict().items() if "backbone" not in k])

    @property
    def trainable_student(self):
        if self.args.train_backbone:
            return self.student
        return self.student.head

    def teacher_dict(self):
        if self.args.train_backbone:
            return self.teacher.state_dict()
        return OrderedDict([(k, v) for k, v in self.teacher.state_dict().items() if "backbone" not in k])

    @property
    def trainable_teacher(self):
        if self.args.train_backbone:
            return self.teacher
        return self.teacher.head


def train_dino(args, writer):
    if not args.disable_ddp:
        utils.init_distributed_mode(args)
    if args.batch_size is not None:
        args.batch_size_per_gpu = args.batch_size // utils.get_world_size()
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    student, _, normalize = load_model(args, split_preprocess=True)
    teacher, _ = load_model(args)

    if not args.precomputed:
        aug = IMAGE_AUGMENTATIONS[args.image_aug](num_augs=args.num_augs, **args.aug_args)
        transform = AugWrapper(
            vit_image_size=args.vit_image_size,
            aug_image_size=args.aug_image_size,
            global_augs=aug,
            normalize=normalize,
            image_size=args.image_size
        )
    else:
        aug = EMBED_AUGMENTATIONS[args.embed_aug](num_augs=args.num_augs, **args.aug_args)
        transform = AugWrapper(
            global_augs=aug
        )

    dataset = getattr(loaders, args.loader)(
        knn_path=args.knn_path,
        datapath=args.datapath,
        k=args.knn,
        transform=transform, dataset=args.dataset,
        precompute_arch=args.arch if args.precomputed else None,
        **args.loader_args)

    if not args.disable_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"In-distribution Data loaded: there are {len(dataset)} images.")
    print("len dataloader", len(data_loader))

    student_teacher_model = TeacherStudentCombo(teacher=teacher, student=student, args=args)
    # move networks to gpu
    student_teacher_model = student_teacher_model.cuda()
    if not args.disable_ddp:
        student_teacher_model = nn.parallel.DistributedDataParallel(student_teacher_model, device_ids=[args.gpu])


    # ============ preparing loss ... ============
    loss_class = getattr(losses, args.loss)
    dino_loss_args = dict(
            out_dim=args.out_dim,
            batchsize=args.batch_size_per_gpu,
            warmup_teacher_temp=args.warmup_teacher_temp,
            teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
            nepochs=args.epochs,
            **args.loss_args)
    if losses.is_multihead(loss_class):
        dino_loss_args.update(num_heads=args.num_heads)
        dino_loss = loss_class(**dino_loss_args).cuda()
    elif args.num_heads == 1:
        dino_loss = loss_class(**dino_loss_args).cuda()
    else:
        dino_loss = nn.ModuleList([loss_class(**dino_loss_args) for _ in range(args.num_heads)]).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student_teacher_model.module.trainable_student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    bs_factor = (args.batch_size_per_gpu * utils.get_world_size()) / 256.
    lr_schedule = utils.cosine_scheduler(
        args.lr * bs_factor,  # linear scaling rule
        args.min_lr * bs_factor,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, args.max_momentum_teacher,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student_teacher_model.module.student,
        teacher=student_teacher_model.module.teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        if not args.disable_ddp:
            data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student_teacher_model, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, writer)

        # ============ writing logs ... ============
        save_dict = {
            'student': student_teacher_model.module.student_dict(),
            'teacher': student_teacher_model.module.teacher_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        try:       
            torch.set_printoptions(profile="full")
            if epoch % 10 == 0:
                d_loss = dino_loss[0] if hasattr(dino_loss, "__getitem__") else dino_loss
                print("highest probs:", torch.topk(d_loss.probs_pos * 100, 50)[0])
                print("lowest probs:", torch.topk(d_loss.probs_pos * 100, 50, largest=False)[0])
            torch.set_printoptions(profile="default")
        except:
            print(" ")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student_teacher_model, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, writer):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images, _ = data
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_out, student_out = student_teacher_model(images)
            if losses.is_multihead(dino_loss) or args.num_heads == 1:
                head_losses = dino_loss(student_out, teacher_out, epoch=epoch)
            else:
                head_losses = torch.stack([d(s, t, epoch=epoch) for d, s, t in zip(dino_loss, student_out, teacher_out)])
            loss = head_losses.mean()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), flush=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student_teacher_model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_teacher_model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student_teacher_model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_teacher_model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            s_head_params = student_teacher_model.module.trainable_student.parameters()
            t_head_params = student_teacher_model.module.trainable_teacher.parameters()
            for param_q, param_k in zip(s_head_params, t_head_params):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update_raw(head_losses=head_losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if utils.is_main_process():
            writer.add_scalar("Train loss step", loss, it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if utils.is_main_process() and args.num_heads > 1:
        avg_loss = metric_logger.meters['head_losses'].global_avg
        student_teacher_model.module.teacher.head.set_losses(avg_loss)
        student_teacher_model.module.student.head.set_losses(avg_loss)

    if utils.is_main_process():
        if args.num_heads == 1:
            writer.add_scalar("Train loss epoch", torch.Tensor([metric_logger.meters['loss'].global_avg]), epoch)
        else:
            avg_loss = metric_logger.meters['head_losses'].global_avg
            writer.add_scalars("Train loss epoch",
                               {f"head{i}": loss for i, loss in enumerate(avg_loss)},
                               epoch)

        d_loss = dino_loss[0] if hasattr(dino_loss, "__getitem__") else dino_loss
        if hasattr(d_loss, 'probs_pos'):
            writer.add_histogram("p(k) over Epochs", d_loss.probs_pos, epoch)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.scalar_meters.items()}


def default_out_dir(loss, dset):
    return f"./experiments/{loss}-{dset}"


def make_out_dir(args):
    if args.output_dir is None:
        args.output_dir = default_out_dir(args.loss, args.loader)
    if args.new_run:
        n = 1
        dir_name = args.output_dir
        while Path(args.output_dir).is_dir():
            n += 1
            args.output_dir = f"{dir_name}{n}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=not args.new_run)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args = process_args(args)

    make_out_dir(args)
    writer = None
    if utils.is_main_process():
        writer = SummaryWriter(args.output_dir)
    with open(os.path.join(args.output_dir, "hp.json"), 'wt') as f:
        json.dump(vars(args), f, indent=4, default=str)
    train_dino(args, writer)


if __name__ == '__main__':
    main()
