# -*- coding: utf-8 -*-
# @time: 2022/5/16 21:47
# @Author: 坤
# @file: __init__.py
# setting the training member, including optimizer, loss ,model
from cifar_noise_data.dataloader import parser
from model.DenseNet169 import DenseNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
from vit_pytorch import ViT
from efficientnet_pytorch import EfficientNet
from model.Inception_v3 import InceptionV3
from model.ResNet import ResNet18
from model.NASNet import NASNet,nasnetmobile

"""
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs',type=int,default=150,help='epochs of training number')
    parser.add_argument('--batch_size',type=int,default=64,help='batch_size of dataloader')
    parser.add_argument('--lr',type=int,default=5e-3,help='learning rate in training rate')
    parser.add_argument('--dataset',type=str,default='cifar100',help='cifar10 or cifar100')
    parser.add_argument('--noise_type',type=str,default='noisy100',help='[clean,worst,random1,random2,random3,noisy100]')
    parser.add_argument('--num_workers',type=int,default=0,help='data num_worker')
    parser.add_argument('--print_freq',type=int,default=200,help='step of print freq')
    parser.add_argument('--network',type=str,default='ViT',help='the network you choise [denseNet,ViT,resnet,Efficientnet,NASnet,InceptionV3]')
    args = parser.parse_args()
    return args
"""
args = parser()
model_name = args.network
num_classes = 100 if args.dataset == 'cifar100' else 10
if args.network == 'DenseNet':
    model = DenseNet(num_classes=100)
elif args.network == 'ViT':
    model = v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1)
elif args.network == 'EfficientNet':
    model = EfficientNet.from_pretrained('efficientnet-b0', False)
elif args.network == 'Inception':
    model = InceptionV3(num_classes=num_classes)
elif args.network == 'ResNet':
    model = ResNet18(num_classes=num_classes)
elif args.network == 'NASNet':
    model = nasnetmobile(num_classes=num_classes,use_aux=False)

from cifar_noise_data.dataloader import Trainer
loss_function = torch.nn.BCELoss()

from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-9)
trainer = Trainer(model=model,
                loss_function=loss_function,
                n_epoch=args.n_epochs,
                noise_type=args.noise_type,
                dataset=args.dataset,
                batch_size=args.batch_size,
                optimizer=optimizer,
                model_name=model_name,
                num_workers=args.num_workers)
trainer.run()
print("{} model training over ...".format(model_name))