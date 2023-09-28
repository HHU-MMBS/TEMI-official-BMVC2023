import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.datasets as tds
from pathlib import Path

import loaders
import model_builders


class PrecomputedEmbeddingDataset(Dataset):

    def __init__(self, dataset, arch, train, datapath):
        super().__init__()
        self.emb, self.targets = model_builders.load_embeds(
            arch=arch,
            dataset=dataset,
            datapath=datapath,
            with_label=True,
            test=not train)

    def __getitem__(self, index):
        return self.emb[index], self.targets[index]

    def __len__(self):
        return len(self.emb)


def get_dataset(dataset, datapath='./data', train=True, transform=None, download=True, precompute_arch=None):
    if precompute_arch:
        return PrecomputedEmbeddingDataset(
            dataset=dataset,
            arch=precompute_arch,
            datapath="data", # assumes embeddings are saved in the ./data folder
            train=train)
    
    load_obj = tds if dataset in ["CIFAR10","CIFAR100", "STL10"] else loaders
    if dataset == "STL10":
        split = 'train' if train else 'test'
        return getattr(load_obj, dataset)(root=datapath,
                        split=split,
                        download=download, transform=transform)
    elif "CIFAR" in dataset:
        return getattr(load_obj, dataset)(root=datapath,
                        train=train,
                        download=download, transform=transform)
    else:
        # imagenet subsets
        # TODO i dont know if val and val_structured are the same
        if "ILSVRC" in datapath and train is False:
            datapath = datapath.replace("train","val")
        return getattr(load_obj, dataset)(root=datapath,
                         transform=transform)


class EmbedNN(Dataset):
    def __init__(self,
                 knn_path,
                 transform,
                 k=10,
                 dataset="CIFAR100",
                 datapath='./data',
                 precompute_arch=None):
        super().__init__()
        self.transform = transform
        self.complete_neighbors = torch.load(knn_path)
        if k < 0:
            k = self.complete_neighbors.size(1)
        self.k = k
        self.neighbors = self.complete_neighbors[:, :k]
        self.datapath = './data' if 'IN' not in dataset else datapath

        self.dataset = get_dataset(
            dataset,
            datapath=datapath,
            transform=None,
            train=True,
            download=True,
            precompute_arch=precompute_arch)

    def get_transformed_imgs(self, idx, *idcs):
        img, label = self.dataset[idx]
        rest_imgs = (self.dataset[i][0] for i in idcs)
        return self.transform(img, *rest_imgs), label

    def __getitem__(self, idx):
        # KNN pair
        pair_idx = np.random.choice(self.neighbors[idx], 1)[0]

        return self.get_transformed_imgs(idx, pair_idx)

    def __len__(self):
        return len(self.dataset)


class TruePosNN(EmbedNN):

    def __init__(self, knn_path, *args, **kwargs):
        super().__init__(knn_path, *args, **kwargs)
        p = Path(knn_path).parent
        nn_p = p / 'hard_pos_nn.pt'
        if nn_p.is_file():
            self.complete_neighbors = torch.load(nn_p)
        else:
            emb = torch.load(p / 'embeddings.pt')
            emb /= emb.norm(dim=-1, keepdim=True)
            d = emb @ emb.T
            labels = torch.tensor(self.dataset.targets)
            same_label = labels.view(1, -1) == labels.view(-1, 1)
            # Find minimum number of images per class
            k_max = same_label.sum(dim=1).min()
            d.fill_diagonal_(-2)
            d[torch.logical_not(same_label)] = -torch.inf
            self.complete_neighbors = d.topk(k_max, dim=-1)[1]
            torch.save(self.complete_neighbors, nn_p)
        self.neighbors = self.complete_neighbors[:, :self.k]
