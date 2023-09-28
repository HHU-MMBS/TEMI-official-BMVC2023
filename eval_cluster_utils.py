"""
Eval ood performance cifar100 -> cifar10
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from torch import linalg
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

import utils
from linear_evaluation import ModelEval
from loaders import get_dataset
from model_builders import load_model


def get_occupied_classes(train_features, plot=True , path='./', return_class_indices=False,
                                name='fig_occupied_classes_box_plot.png'):
    class_weights = train_features.mean(dim=0)
    # avg class probability
    threshhold = class_weights.mean()

    occupied_classes_idx = (class_weights> (threshhold))
    occupied_classes = occupied_classes_idx.sum()
    if plot:
        sorted_np_array = np.sort(class_weights.cpu().view(-1).numpy())[-occupied_classes:]
        ids = [i for i in range(sorted_np_array.shape[0])]
        _ = plt.bar(ids,sorted_np_array, label=name)
        plt.legend()
        #plt.savefig(os.path.join(path, name))

    return occupied_classes, occupied_classes_idx


def pk_statistics(features, config):
    pkx = (features / config.teacher_temp).softmax(dim=-1)
    labels = pkx.argmax(dim=-1)
    occupied_classes = len(torch.unique(labels))
    entropy = -(pkx * pkx.log()).sum(dim=-1).mean()
    pk = pkx.mean(dim=0)
    return pk, entropy, occupied_classes


def _kl_div(p, q):
    return (p * (p / q).log()).sum()


def _jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def jsd_to_train(pk_train, pk_test, pk_test_ood):
    return _jsd(pk_train, pk_test), _jsd(pk_train, pk_test_ood)


def compute_spmi(train_features, test_features, config):
    pkx = (train_features / config.teacher_temp).softmax(dim=-1)
    pk = pkx.mean(dim=0)
    pkx_test = (test_features / config.teacher_temp).softmax(dim=-1)
    spmi = (pkx**2 / pk).sum(dim=-1).mean()
    spmi_test = (pkx_test**2 / pk).sum(dim=-1).mean()
    return spmi.item(), spmi_test.item()


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels).cuda()
        test_labels = torch.from_numpy(test_labels).cuda()

    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()

        temp = torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            )

        probs = torch.sum(temp,1)
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class FeatureExtractionPipeline:
    def __init__(self, args, cache_backbone=False, datapath='./data'):
        if not args.head and cache_backbone:
            raise ValueError("head must be True if cache_backbone is True")
        self.args = args
        self.cache_backbone = cache_backbone
        self.embeds = None

        self.model, transform = load_model(self.args, head=args.head)
        self.model.cuda().eval()
        if not self.cache_backbone:
            self.model = nn.DataParallel(self.model)
        if args.lin_eval:
            backbone = self.model
            embed_dim = utils.embed_dim(self.args, self.model) 
            num_classes = 100 if args.dataset=='CIFAR100' else 10
            self.model = ModelEval(backbone, embed_dim, args.hidden_dim, args.bottleneck_dim,
                                    num_classes,args.nlayers,args.linear_only)
                
        precompute_arch = args.arch if args.precomputed else None
        dataset_train = get_dataset(args.dataset, datapath=datapath,
                                    train=True,
                                    download=True, transform=transform,
                                    precompute_arch=precompute_arch)
        dataset_val = get_dataset(args.dataset, datapath=datapath,
                                  train=False,
                                  download=True, transform=transform,
                                  precompute_arch=precompute_arch)

        for ds_name, dataset in zip(["dataset_labels", "val_labels", "test_labels"],
                                    [dataset_train, dataset_val]):
            if dataset is not None:
                try:
                    setattr(self, ds_name, np.array(dataset.targets, dtype=np.int64))
                except AttributeError:
                    setattr(self, ds_name, np.array(dataset.labels, dtype=np.int64))
            else:
                setattr(self, ds_name, None)

        batch_size = args.batch_size_per_gpu or args.batch_size
        self.data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )


    @property
    def cached(self):
        return self.embeds is not None

    @torch.no_grad()
    def get_features(self, pretrained_weights):
        if isinstance(self.model, nn.DataParallel):
            module = self.model.module
        else:
            module = self.model
        if self.args.lin_eval:
            utils.load_pretrained_weights(module,
                                          pretrained_weights,
                                          self.args.checkpoint_key,
                                          head=True,
                                          head_only=False)
        else:
            utils.load_pretrained_weights(module,
                                          pretrained_weights,
                                          self.args.checkpoint_key,
                                          head=self.args.head,
                                          head_only=self.cached)
        self.model.eval()

        if not self.cache_backbone:
            train_features = extract_features(self.model, self.data_loader_train, self.args.head)
            test_features = extract_features(self.model, self.data_loader_val, self.args.head)
        else:
            embeds = self.get_embeds()
            train_features = self.head_features(embeds['train'])
            test_features = self.head_features(embeds['test'])

        print(f"Train feats {train_features.shape}")
        return train_features, test_features, self.dataset_labels, self.val_labels

    @torch.no_grad()
    def head_features(self, embed):
        features = []
        for samples in embed:
            samples = samples.cuda()
            features.append(self.model.head(samples).cpu())
        return torch.cat(features)

    @torch.no_grad()
    def get_embeds(self):
        if self.embeds is not None:
            print('Using cached embedding')
            return self.embeds
        print('Compute embeddings')
        embeds = defaultdict(list)
        for k, loader in zip(['train', 'test'],
                             [self.data_loader_train, self.data_loader_val]):
            if loader is None:
                continue
            for samples, _ in tqdm(loader):
                if not isinstance(samples, list):
                    samples = [samples]
                samples = torch.cat([im.cuda(non_blocking=True) for im in samples])
                output = self.model.backbone_embed(samples)
                embeds[k].append(output.cpu())
        self.embeds = embeds
        return embeds


@torch.no_grad()
def extract_features(model, data_loader, head):
    features = []
    for (samples, _) in tqdm(data_loader):
        if not isinstance(samples, list):
            samples = [samples]
        samples = torch.cat([im.cuda(non_blocking=True) for im in samples])
        if head:
            try:
                feats, _ = model(samples)
            except Exception:
                feats = model(samples)
        else:
            feats = model(samples)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


def norm_by_name(x, norm):
    if norm == "softmax":
        return x.softmax(dim=-1)
    elif norm == "l2":
        return torch.nn.functional.normalize(x, dim=-1, p=2)
    elif norm == "l1":
        return torch.nn.functional.normalize(x, dim=-1, p=1)
    return x


def norm_feats(*features, norm="softmax"):
    return [norm_by_name(x, norm) for x in features]

@torch.no_grad()
def calc_maha_distance(embeds, means_c, inv_cov_c):
    diff = embeds - means_c
    #dist = np.matmul(np.matmul(diff, inv_cov_c),diff.T)
    dist = np.matmul(diff,inv_cov_c)*diff
    dist = np.sum(dist,axis=1)
    return dist

@torch.no_grad()
def OOD_classifier_maha(train_embeds_in, train_labels_in, test_embeds_in, test_embeds_outs, num_classes,
                        relative=False, std_all=False):
    class_covs = []
    class_means = []
    used_classes = 0
    # calculate class-wise means and covariances
    for c in range(num_classes):
        train_embeds_c = train_embeds_in[np.where(train_labels_in == c)]
        if len(train_embeds_c)>1:
            class_mean = np.mean(train_embeds_c, axis = 0)
            cov = np.cov((train_embeds_c - (class_mean.reshape([1,-1]))).T )
            class_covs.append(cov)
            class_means.append(class_mean)
            used_classes += 1


    # class-wise std estimation
    if not std_all:
        cov_invs = np.linalg.inv(np.mean(np.stack(class_covs, axis=0),axis=0))
    else:
        # estimating the global std from train data
        avg_train_mean = np.mean(train_embeds_in, axis=0)
        cov_invs =  np.linalg.inv(np.cov((train_embeds_in-avg_train_mean.reshape([1,-1])).T))

    scores_in_dist = [calc_maha_distance(test_embeds_in, class_means[c], cov_invs) for c in range(used_classes)]
    scores_out_dist = [calc_maha_distance(test_embeds_outs, class_means[c], cov_invs) for c in range(used_classes)]

    # classes X num_data
    scores_in_dist = np.stack(scores_in_dist)
    scores_out_dist = np.stack(scores_out_dist)

    if relative == True:
        avg_train_mean = np.mean(train_embeds_in, axis=0)
        avg_train_inv_cov =  np.linalg.inv(np.cov((train_embeds_in-avg_train_mean.reshape([1,-1])).T))
        avg_train_score_in = calc_maha_distance(test_embeds_in, avg_train_mean , avg_train_inv_cov )
        avg_train_score_out = calc_maha_distance(test_embeds_outs, avg_train_mean , avg_train_inv_cov)
        scores_in_dist -= avg_train_score_in
        scores_out_dist -= avg_train_score_out

    # Get OOD score for each datapoint
    scores_in_dist = -np.min(scores_in_dist, axis=0)
    scores_out_dist = -np.min(scores_out_dist, axis=0)

    scores = np.concatenate([scores_in_dist, scores_out_dist])
    return scores


def get_eval_args(notebook=False):
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')
    parser.add_argument('--temperature', default=0.02, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--out_dim', default=100, type=int)
    parser.add_argument('--head', default=True, type=utils.bool_flag,
        help="Whether to load the DINO head")
    parser.add_argument('--vit_image_size', type=int, default=224, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--datapath', default='./data', type=str)

    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', "STL10", \
                                                                "CIFAR20", "IN1K", "IN50", 'IN100', "IN200"], type=str)
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--image_size', type=int, default=32, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--crops_scale', type=float, nargs='+', default=(0.8, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping.""")
    parser.add_argument('--crops_number', type=int, default=1, help="""Number of
        local views to generate. Set this parameter to 0 to disable multi-crop training.""")

    parser.add_argument('--nlayers', default=2, type=int, help='Head layers')
    parser.add_argument('--hidden_dim', default=512, type=int, help="Head's hidden dim")
    parser.add_argument('--bottleneck_dim', default=256, type=int, help="Head's bottleneck dim")
    parser.add_argument('--l2_norm', default=False, help="Whether to apply L2 norm after backbone") 
    parser.add_argument('--ckpt_folder', type=str) 
    parser.add_argument('--no_cache', action='store_true', default=False, help='Whether to cache backbone results')
    parser.add_argument('--ignore_hp_file', action='store_true', default=False, help='Whether to ignore hp.json')
    parser.add_argument('--lin_eval', default=False, help='True if the model has a backbone and linear layer.')
    parser.add_argument('--linear_only', default=False, type=utils.bool_flag, help='True if head is only a linear layer.')
    return parser.parse_args() if notebook is False else parser.parse_args("") 
