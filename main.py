#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/05/21 20:29:58
@Author  :   ykzhou 
@Version :   0.0
@Contact :   ykzhou@stu.xidian.edu.cn
@Desc    :   None
'''
#%%
from tkinter import _Padding
import torchvision.transforms as transforms
from cifar_noise_data.cifar import CIFAR10
import numpy as np
from cifar_noise_data.datasets import input_dataset 
#%%
# trainset,a,b,c = input_dataset(dataset='cifar10',noise_path='./cifar_noise_data/CIFAR-10_human.pt',noise_type='worse_label',is_human=False,model_name=None)

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform2 = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = CIFAR10(root='./cifar_noise_data',train=True,
                  noise_path='./cifar_noise_data/CIFAR-10_human.pt',noise_type='worse_label',
                  download=False)  

dataset2 = CIFAR10(root='./cifar_noise_data',train=True,transform=transform2,noise_path='./cifar_noise_data/CIFAR-10_human.pt',
                   noise_type='worse_label',download=False)