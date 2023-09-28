import os
import numpy as np
from pathlib import Path
import json

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import argparse

from model_builders import load_model
import utils
from loaders import get_dataset
from utils import embed_dim
from utils import trunc_normal_
        
class ModelEval(nn.Module):
    def __init__(self, backbone, in_dim, hidden_dim, 
                        bottleneck_dim, num_classes, num_layers, linear_only,
                        train_backbone=False, l2_norm=False):
        super(ModelEval, self).__init__()
        self.train_backbone = train_backbone
        self.linear_only = linear_only
        self.out_dim = num_classes
        self.l2_norm = l2_norm
       
        if linear_only:
            self.mlp = nn.Linear(in_dim, num_classes)
            self.mlp.weight.data.normal_(mean=0.0, std=0.01)
            self.mlp.bias.data.zero_()
        else:
            self.mlp = self.init_mlp(in_dim, hidden_dim, num_layers, bottleneck_dim)
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def init_mlp(self, in_dim, hidden_dim, num_layers, bottleneck_dim):
        if num_layers == 1:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        for layer in layers:
            layer.apply(self._init_weights)
        last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, self.out_dim, bias=False))
        last_layer.weight_g.data.fill_(1)
        layers.append(last_layer)
        mlp = nn.Sequential(*layers)
        return mlp

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        x = self.backbone(x).detach() if not self.train_backbone else self.backbone(x)
        if self.l2_norm:
            x = F.normalize(x, dim=1, p=2)
        return self.mlp(x) 
        
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_iters=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_iters
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def validate(model, val_loader, device, targets):
    correct = 0
    total = 0
    loss_step = []
    max_indx_pred = []
    with torch.no_grad():
        for inp_data, labels in val_loader:
            labels = labels.to(device)
            inp_data = inp_data.to(device)
            outputs = model(inp_data)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            max_indx_pred.append(predicted.cpu())
    val_acc = (100 * correct / total).cpu().numpy()
    preds = torch.cat(max_indx_pred).numpy()
    cluster_acc, nmi, anmi, ari = utils.compute_metrics(targets, preds, min_samples_per_class=5)
    return val_acc, cluster_acc, nmi, anmi, ari

def apply_color_distortion(s=0.5):  
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def get_data_loaders(args, normalize):
    datapath = './data' if  args.dataset in ["CIFAR10", "CIFAR100", "STL10", "CIFAR20"] else args.datapath
    
    if not args.weak_augs:
        transform_train = transforms.Compose([
                         transforms.Resize(224, interpolation=3),
                         normalize,
                         ])
    else:
        transform_train = transforms.Compose([
                         transforms.RandomResizedCrop(224, scale=(0.5, 1), interpolation=3),
                         transforms.Resize(224, interpolation=3),
                         transforms.RandomHorizontalFlip(),
                         apply_color_distortion(s=0.2),
                         normalize,
                         ])
    
    transform_test = transforms.Compose([
                     transforms.Resize(224,interpolation=3),
                     normalize,
                     ])
    
    dataset_train = get_dataset(args.dataset, datapath=datapath,
                        train=True,
                        download=True, transform=transform_train)
    dataset_val = get_dataset(args.dataset, datapath=datapath,
                        train=False,
                        download=True, transform=transform_test)

    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)
    
    val_loader = DataLoader(dataset_val,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=4)   
    try:
        val_labels = np.array(dataset_val.targets,dtype=np.int64)
    except:
        val_labels = np.array(dataset_val.labels,dtype=np.int64)
    
    num_classes = len(np.unique(val_labels))
    
    return train_loader, val_loader, num_classes, val_labels
    
        
def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    losses = []
    criterion = nn.CrossEntropyLoss()

    for it, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, labels = data
        labels = labels.to(device)
        images = images.to(device)
        it = len(train_loader) * ep + it 
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = scheduler[it]
        logits = model(images) 
        loss = criterion(logits, labels)        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
            
    avg_loss = torch.tensor(losses).mean().numpy()
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training MLP on top of frozen backbone (supervised)')
    parser.add_argument('--load_path', default='', type=str, help='Path to pretrained weights.')
    parser.add_argument('--checkpoint_key', default='teacher', type=str, help='Key to use in the checkpoint.')
    parser.add_argument('--pretrained_weights', default=None, type=str, help='Path to pretrained model weights. ')
    parser.add_argument('--save_path', default='./experiments/finetune/', type=str, help='Path to save model checkpoint.')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', "STL10", \
                                                                "CIFAR20", "IN1K", "IN50", 'IN100', "IN200"], type=str)
    parser.add_argument('--datapath', default='./data', type=str)
    parser.add_argument('--batch_size', type=int, default=256, help="""Value for batch size.""")
    parser.add_argument('--lr', type=float, default=5e-3, help="""Value for learning rate.""")
    parser.add_argument('--wd', type=float, default=1e-2, help="""Value for weight decay.""")
    parser.add_argument('--num_epochs', type=int, default=100, help="""Number of training epochs.""")
    parser.add_argument('--arch', default='dino_vitb16', help="""Chosen architecture for backbone.""")
    parser.add_argument('--vit_image_size', type=int, default=224, help="""Size of images for VIT.""")
    parser.add_argument('--hidden_dim', type=int, default=512, help="""Hidden dimension in MLP.""")
    parser.add_argument('--bottleneck_dim', type=int, default=256, help="""Dimension of bottleneck in MLP.""")
    parser.add_argument('--num_layers', type=int, default=2, help="""Number of layers in MLP.""")
    parser.add_argument('--linear_head', type=bool, default=True, help="""True if head should only be a linear layer instead of MLP.""")
    parser.add_argument('--train_backbone', type=bool, default=False, help="""True if also the backbone should be trained.""")
    parser.add_argument('--l2_norm', type=bool, default=False, help="""Whether to apply L2 normalization to the output of the backbone.""")
    parser.add_argument('--weak_augs', default=False, help="""Whether to apply augmentations or not.""")
    
    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, args.dataset, f'exp_v002_adam_MLP_rrc05_{str(np.random.randint(1000)).zfill(4)}')
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "hp.json", 'wt') as f:
        json.dump(vars(args), f, indent=4, default=str)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    backbone, _ , normalize = load_model(args, head=False, split_preprocess=True)
    train_loader, val_loader, num_classes, val_labels = get_data_loaders(args, normalize)
    
    if args.pretrained_weights != None: 
        utils.load_pretrained_weights(backbone,
                                      args.pretrained_weights,
                                      args.checkpoint_key,
                                      head=True,
                                      head_only=False)

    backbone = backbone.to(device)
    backbone_dim = embed_dim(args, backbone)
    
    model = ModelEval(backbone, backbone_dim, args.hidden_dim, 
                args.bottleneck_dim, num_classes, args.num_layers, args.linear_head,
                args.train_backbone)
    model = model.to(device)
    if not args.train_backbone:
        print('Training the head only \n\n')
        model.backbone.eval()
        # TODO try simple SGD here
        optimizer = torch.optim.Adam(model.mlp.parameters(), 
                        lr = args.lr,
                        weight_decay=args.wd)
    else:
        print('Training the whole model \n\n')
        optimizer = torch.optim.Adam(model.parameters(), 
                        lr = args.lr,
                        weight_decay=args.wd)
    
    lr_schedule = cosine_scheduler(args.lr, 0,args.num_epochs,len(train_loader), warmup_iters=0, start_warmup_value=0)
    
    max_val_acc = 0
    for ep in range(args.num_epochs):
        avg_loss = train_one_epoch(model= model,
                                   train_loader=train_loader, 
                                   optimizer=optimizer, 
                                   scheduler = lr_schedule,
                                   device = device)
        
        with torch.no_grad():
            val_acc, cluster_acc, nmi, anmi, ari = validate(model, val_loader, device, val_labels)
            if val_acc > max_val_acc:
                torch.save(model.state_dict(), output_dir / 'best_model.pth')
                max_val_acc = val_acc
                best_dict_data = {
                    "val_acc" : float(val_acc),
                    "cluster_acc" : float(cluster_acc),
                    "NMI" : nmi,
                    "ARI" : anmi,
                    "ANMI" : ari,
                    "epoch": ep}
                with open(output_dir / "best-results.json", 'w') as f:
                    json.dump(best_dict_data, f, indent=4)
            
        print(f'Epoch {ep} Average training loss: {avg_loss:.3}, Validation accuracy {val_acc:.3}, \
            Cluster Acc {cluster_acc:.2} Maximum Val acc: {max_val_acc:.2}')
    
    # Compute clustering metrics and save them to a file
    with open(output_dir / "best-results.json", 'w') as f:
        json.dump(best_dict_data, f, indent=4)
