""" helper function

author baiyu
"""
import os
import sys
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def get_network(netname):
    """ return given network
    """
    if netname == 'vgg16':
        from .architectures.vgg import vgg16_bn
        net = vgg16_bn()
    elif netname == 'vgg13':
        from .architectures.vgg import vgg13_bn
        net = vgg13_bn()
    elif netname == 'vgg11':
        from .architectures.vgg import vgg11_bn
        net = vgg11_bn()
    elif netname == 'vgg19':
        from .architectures.vgg import vgg19_bn
        net = vgg19_bn()
    elif netname == 'densenet121':
        from .architectures.densenet import densenet121
        net = densenet121()
    elif netname == 'densenet161':
        from .architectures.densenet import densenet161
        net = densenet161()
    elif netname == 'densenet169':
        from .architectures.densenet import densenet169
        net = densenet169()
    elif netname == 'densenet201':
        from .architectures.densenet import densenet201
        net = densenet201()
    elif netname == 'googlenet':
        from .architectures.googlenet import googlenet
        net = googlenet()
    elif netname == 'googlenet_features':
        from .architectures.googlenet import googlenet_features
        net = googlenet_features()
    elif netname == 'inceptionv3':
        from .architectures.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif netname == 'inceptionv4':
        from .architectures.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif netname == 'inceptionresnetv2':
        from .architectures.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif netname == 'xception':
        from .architectures.xception import xception
        net = xception()
    elif netname == 'resnet18':
        from .architectures.resnet import resnet18
        net = resnet18()
    elif netname == 'resnet32':
        from .architectures.resnet_s import resnet32_cifar100
        net = resnet32_cifar100()
    elif netname == 'resnet34':
        from .architectures.resnet import resnet34
        net = resnet34()
    elif netname == 'resnet50':
        from .architectures.resnet import resnet50
        net = resnet50()
    elif netname == 'resnet101':
        from .architectures.resnet import resnet101
        net = resnet101()
    elif netname == 'resnet152':
        from .architectures.resnet import resnet152
        net = resnet152()
    elif netname == 'preactresnet18':
        from .architectures.preactresnet import preactresnet18
        net = preactresnet18()
    elif netname == 'preactresnet34':
        from .architectures.preactresnet import preactresnet34
        net = preactresnet34()
    elif netname == 'preactresnet50':
        from .architectures.preactresnet import preactresnet50
        net = preactresnet50()
    elif netname == 'preactresnet101':
        from .architectures.preactresnet import preactresnet101
        net = preactresnet101()
    elif netname == 'preactresnet152':
        from .architectures.preactresnet import preactresnet152
        net = preactresnet152()
    elif netname == 'resnext50':
        from .architectures.resnext import resnext50
        net = resnext50()
    elif netname == 'resnext101':
        from .architectures.resnext import resnext101
        net = resnext101()
    elif netname == 'resnext152':
        from .architectures.resnext import resnext152
        net = resnext152()
    elif netname == 'shufflenet':
        from .architectures.shufflenet import shufflenet
        net = shufflenet()
    elif netname == 'shufflenetv2':
        from .architectures.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif netname == 'squeezenet':
        from .architectures.squeezenet import squeezenet
        net = squeezenet()
    elif netname == 'mobilenet':
        from .architectures.mobilenet import mobilenet
        net = mobilenet()
    elif netname == 'mobilenetv2':
        from .architectures.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif netname == 'nasnet':
        from .architectures.nasnet import nasnet
        net = nasnet()
    elif netname == 'attention56':
        from .architectures.attention import attention56
        net = attention56()
    elif netname == 'attention92':
        from .architectures.attention import attention92
        net = attention92()
    elif netname == 'seresnet18':
        from .architectures.senet import seresnet18
        net = seresnet18()
    elif netname == 'seresnet34':
        from .architectures.senet import seresnet34
        net = seresnet34()
    elif netname == 'seresnet50':
        from .architectures.senet import seresnet50
        net = seresnet50()
    elif netname == 'seresnet101':
        from .architectures.senet import seresnet101
        net = seresnet101()
    elif netname == 'seresnet152':
        from .architectures.senet import seresnet152
        net = seresnet152()
    elif netname == 'wideresnet':
        from .architectures.wideresidual import wideresnet
        net = wideresnet()
    elif netname == 'stochasticdepth18':
        from .architectures.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif netname == 'stochasticdepth34':
        from .architectures.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif netname == 'stochasticdepth50':
        from .architectures.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif netname == 'stochasticdepth101':
        from .architectures.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    elif netname == 'stochasticdepth110':
        from .architectures.resnet_sd import createModel
        net = createModel(depth = 110, num_classes = 100, death_mode='linear', data='CIFAR100')
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net

class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def get_train_val_dataloader(transform_train=None, batch_size=16, num_workers=2, val_prop=0.1, random_seed=42, dataloader_generator_seed=42,
trainsize=None, mean=CIFAR100_MEAN, std=CIFAR100_STD, val_batchsize=None, eval_trainloader=False, data_path='/home/pedro-pordeus/data'):
    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    if transform_train is None:
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

   
    full_cifar_trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True)
    g = torch.Generator()   
    g.manual_seed(dataloader_generator_seed)
    
    if val_prop == 0:
        trainset = ApplyTransform(full_cifar_trainset, transform_train)
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=g, num_workers=num_workers)
    
    if val_prop > 0:
        indices = np.arange(len(full_cifar_trainset))
        train_idx, val_idx = train_test_split(indices, test_size=val_prop, shuffle=True, stratify=full_cifar_trainset.targets, random_state=random_seed)
        if trainsize:
            targets = np.array(full_cifar_trainset.targets)
            samples_per_class = trainsize/100
            original_samples_per_class = len(train_idx)/100
            train_idx = train_idx[np.argsort(targets[train_idx])]
            train_idx = np.hstack([train_idx[int(i*original_samples_per_class):int(i*original_samples_per_class+samples_per_class)] for i in range(100)])
        trainset = ApplyTransform(torch.utils.data.Subset(full_cifar_trainset, train_idx), transform_train)
        valset = ApplyTransform(torch.utils.data.Subset(full_cifar_trainset, val_idx), transform_val)

        cifar100_training_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=g, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        cifar100_validation_loader = torch.utils.data.DataLoader(valset, batch_size=val_batchsize if val_batchsize else len(valset), shuffle=False,num_workers=num_workers, pin_memory=True)
        if eval_trainloader: 
            _, eval_train_idx = train_test_split(train_idx, test_size=val_prop*(len(indices))/len(train_idx), shuffle=True, stratify=np.array(full_cifar_trainset.targets)[train_idx], random_state=random_seed)
            eval_trainset = ApplyTransform(torch.utils.data.Subset(full_cifar_trainset, eval_train_idx), transform_val)
            cifar100_training_evalloader = torch.utils.data.DataLoader(eval_trainset, batch_size=val_batchsize if val_batchsize else len(eval_trainset), shuffle=False, num_workers=num_workers)
            return cifar100_training_loader, cifar100_validation_loader, cifar100_training_evalloader
        return cifar100_training_loader, cifar100_validation_loader


def get_test_dataloader(batch_size=16, num_workers=2, shuffle=False, mean=CIFAR100_MEAN, std=CIFAR100_STD, root='/home/pedro-pordeus/data'):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_test_loader
