#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   detection.py
@Time    :   2022/05/19 21:26:43
@Author  :   ykzhou 
@Version :   0.0
@Contact :   ykzhou@stu.xidian.edu.cn
@Desc    :   None
'''
import torch
from model.O2U_net import O2U_net
from model.ResNet import ResNet18
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--datasets',type=str,default='cifar100')
noise_type = parser.add_argument('--noise_type',type=str,default='noisy_label')
args = parser.parse_args()
num_classes = 100 if args.datasets == 'cifar100' else 10
model = ResNet18(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=device)
print(f"datasets: {args.datasets}, noise_type: {args.noise_type}, n_epochs: {2},")
o2u_net = O2U_net(datasets=args.datasets,noise_type=args.noise_type,n_epochs=2,batch_size=64,model=model,max_epochs=2,num_workers=0,circle_length=2)
o2u_net.detection()