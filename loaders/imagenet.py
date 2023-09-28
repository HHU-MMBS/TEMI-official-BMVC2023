"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
copied and adjusted from https://raw.githubusercontent.com/wvangansbeke/Unsupervised-Classification/master/data/imagenet.py
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf
from glob import glob


class ImageNet(datasets.ImageFolder):
    def __init__(self, root='', split='train', transform=None):
        super(ImageNet, self).__init__(root=root,
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize((256,256))
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img


class ImageNetSubset(data.Dataset):
    def __init__(self, subset_file, root='/home/shared/DataSets/ILSVRC/Data/CLS-LOC/train', split='train', 
                    transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i)) 
        self.imgs = imgs 
        self.classes = class_names
        self.targets = list(zip(*self.imgs))[-1]
	    # Resize
        self.resize = tf.Resize((256,256))

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        #out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}
        return img,target



def IN50(**kwargs):
    return ImageNetSubset(subset_file="loaders/imagenet_subsets/imagenet_50.txt",**kwargs)

def IN100(**kwargs):
    return ImageNetSubset(subset_file="loaders/imagenet_subsets/imagenet_100.txt",**kwargs)

def IN200(**kwargs):
    return ImageNetSubset(subset_file="loaders/imagenet_subsets/imagenet_200.txt",**kwargs)


def IN1K(**kwargs):
    return datasets.ImageFolder(**kwargs)