from enum import Enum
from typing import Union, Tuple, Callable, List, Optional

import PIL
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class InputType(Enum):
    PIL_IMAGE = 1
    TENSOR_IMAGE = 2
    EMBEDDING = 3


class AugmentationBaseClass:

    def __init__(self, trafo, num_augs=1, input_type : InputType = InputType.PIL_IMAGE):
        super().__init__()
        self.input_type = input_type
        self.trafo = trafo
        self.num_augs = num_augs

    def transform(self, image):
        return self.trafo(image)

    def __call__(self, x):
        if self.input_type == InputType.TENSOR_IMAGE:
            x = TF.to_tensor(x)
        aug_x = [self.transform(x) for _ in range(self.num_augs)]
        if self.input_type == InputType.PIL_IMAGE:
            aug_x = [TF.to_tensor(xi) for xi in aug_x]
        return aug_x

    @staticmethod
    def init_from_trafo(trafo, input_type=InputType.PIL_IMAGE):
        def init_fn(num_augs):
            return AugmentationBaseClass(trafo, num_augs=num_augs, input_type=input_type)
        return init_fn

    @staticmethod
    def init_from_trafo_init(trafo_init, input_type=InputType.PIL_IMAGE, defaults=None):
        if defaults is None:
            defaults = {}
        defaults = defaults.copy()

        def init_fn(num_augs, **kwargs):
            kwargs = {**defaults, **kwargs}
            return AugmentationBaseClass(trafo_init(**kwargs), num_augs=num_augs,
                                         input_type=input_type)
        return init_fn


aug_type = Callable[[PIL.Image.Image], List[torch.Tensor]]
img_size_type = Union[int, Tuple[int, int]]


class AugWrapper:

    def __init__(self,
                 global_augs: aug_type,
                 vit_image_size: Optional[img_size_type] = None,
                 normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 local_augs: Optional[aug_type] = None,
                 aug_image_size: Optional[img_size_type] = None,
                 image_size: Optional[img_size_type] = None,
                 train=True):
        if local_augs is not None:
            raise NotImplementedError('Local augs not implemented')
        self.normalize = normalize
        self.global_augs = global_augs
        if vit_image_size is not None:
            image_size = image_size if image_size is not None else vit_image_size
            aug_image_size = aug_image_size if aug_image_size is not None else vit_image_size
            self.aux_resize = transforms.Resize(image_size)
            self.aug_resize = transforms.Resize(aug_image_size)
            if train:
                self.crop = transforms.RandomCrop(vit_image_size)
            else:
                self.crop = transforms.CenterCrop(vit_image_size)
        else:
            self.crop = None
            self.aug_resize = None
            self.aux_resize = None

    def __call__(self, image1, image2, image3=None):
        if self.aux_resize is not None and image3 is not None:
            image3 = self.aux_resize(image3)
        imgs = [img for img in [image1, image2, image3] if img is not None]
        augs =[]
        if self.aug_resize is not None:
            imgs = [self.aug_resize(img) for img in imgs]
        for img in imgs:
            augs.extend(self.global_augs(img))
        if self.crop is not None:
            augs = [self.crop(img) for img in augs]
        if self.normalize is not None:
            augs = [self.normalize(img) for img in augs]
        return augs


def _auto_aug_init(policy):
    policy = getattr(transforms.AutoAugmentPolicy, policy)
    return transforms.AutoAugment(policy=policy)


def _gaussian_noise(std):
    def trafo(x):
        return x + torch.randn_like(x) * std
    return trafo


IMAGE_AUGMENTATIONS = {
    'randaug': AugmentationBaseClass.init_from_trafo_init(transforms.RandAugment,
                                                          defaults={'num_ops': 9, 'magnitude': 1}),
    'autoaug': AugmentationBaseClass.init_from_trafo_init(_auto_aug_init, defaults={'policy': 'IMAGENET'}),
    'mild': AugmentationBaseClass.init_from_trafo(transforms.Compose([
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip()
    ])),
    'gaussian': AugmentationBaseClass.init_from_trafo_init(
        _gaussian_noise, defaults={'std': 0.01}, input_type=InputType.TENSOR_IMAGE),
    'none': lambda *_, **__: lambda *x: [TF.to_tensor(xi) for xi in x]
}

EMBED_AUGMENTATIONS = {
    'gaussian': AugmentationBaseClass.init_from_trafo_init(
        _gaussian_noise, defaults={'std': 0.1}, input_type=InputType.EMBEDDING),
    'none': lambda *_, **__: lambda *x: x
}
