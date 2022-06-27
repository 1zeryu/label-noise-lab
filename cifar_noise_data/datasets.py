import numpy as np 
import torchvision.transforms as transforms
from cifar_noise_data.cifar import CIFAR10, CIFAR100


Inception_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
def input_dataset(dataset, noise_type, noise_path, is_human,model_name):
    if dataset == 'cifar10':
        train_transform = train_cifar10_transform
        test_transform = test_cifar10_transform
    else:
        train_transform = train_cifar100_transform
        test_transform = test_cifar100_transform
    if model_name == 'Inception':
        train_transform = Inception_transform
        test_transform = Inception_transform
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='./cifar_noise_data/',
                                download=True,  
                                train=True, 
                                transform = train_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human
                           )
        test_dataset = CIFAR10(root='./cifar_noise_data/',
                                download=False,  
                                train=False, 
                                transform = test_transform,
                                noise_type=noise_type
                          )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='./cifar_noise_data/',
                                download=True,  
                                train=True, 
                                transform=train_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )
        test_dataset = CIFAR100(root='./cifar_noise_data/',
                                download=False,  
                                train=False, 
                                transform=test_transform,
                                noise_type=noise_type
                            )
        num_classes = 100
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples








