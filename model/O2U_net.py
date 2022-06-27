from distutils.archive_util import make_archive
import os
from time import time
from docutils import TransformSpec
# from more_itertools import value_chain
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import torchvision.transforms as transforms
# from model import cotnet
import datasets
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets import input_dataset
from PIL import Image
from ResNet import ResNet18
from tqdm import tqdm, trange
from cifar import CIFAR10, CIFAR100
from torch.utils.tensorboard import SummaryWriter

Augumentation_transforms1 = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.20100))
])

Augumentation_transforms2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


class Mask_Select(Dataset):
    def __init__(self, origin_dataset, mask_index):
        self.transform = origin_dataset.transform
        self.target_transform = origin_dataset.target_transform
        labels = origin_dataset.train_noisy_labels
        dataset = origin_dataset.train_data
        self.dataname = origin_dataset.dataset
        self.origin_dataset = origin_dataset
        self.train_data = []
        self.train_noisy_labels = []
        for i, m in enumerate(mask_index):
            if m < 0.5:
                continue
            self.train_data.append(dataset[i])
            self.train_noisy_labels.append(labels[i])

        print("origin set number:%d" % len(labels), "after clean number:%d" % len(self.train_noisy_labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_noisy_labels[index]

        if self.dataname != 'MinImagenet':
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)


class Plot_graph():
    def __init__(self, model_name, datasets, noise_type):
        datatime = time.strftime('%Y%m%d%H%M')
        path = model_name + "_" + datasets + noise_type + "_" + datatime
        self.writer = SummaryWriter('./runs/' + path + '/')

    def loss_graph(self, epoch, value, model_name, stage):
        self.writer.add_scalar('{}/{}/Loss'.format(model_name, stage), value, epoch)

    def acc_graph(self, epoch, test_acc, model_name, stage):
        self.writer.add_scalar('{}/{}/test'.format(model_name, stage), test_acc, epoch)

    def noise_acc(self, epoch, value, model_name):
        self.writer.add_scalar('{}/noise_acc'.format(model_name), value, epoch)

    def first_stage_graph(self, epoch, loss, acc, model_name):
        self.loss_graph(epoch=epoch, value=loss, model_name=model_name, stage='first_stage')
        self.acc_graph(epoch=epoch, test_acc=acc, model_name=model_name, stage='first_stage')

    def third_stage_graph(self, epoch, loss, acc, model_name):
        self.loss_graph(epoch=epoch, value=loss, model_name=model_name, stage='third_stage')
        self.acc_graph(epoch=epoch, test_acc=acc, model_name=model_name, stage='third_stage')

    def second_stage_graph(self, epoch, loss, acc, model_name):
        self.loss_graph(epoch=epoch, value=loss, model_name=model_name, stage='third_stage')
        self.acc_graph(epoch=epoch, test_acc=acc, model_name=model_name, stage='third_stage')

    def close(self):
        self.writer.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class O2U_net():
    def __init__(self, datasets, noise_type, n_epochs, batch_size, model, num_workers=0,
                 forget_rate=None, model_name='ResNet', max_epochs=250, lr=0.01, circle_length=1) -> None:
        self.max_epochs = max_epochs
        self.datasets = datasets
        self.num_workers = num_workers
        self.n_epochs = n_epochs
        self.noise_type = noise_type
        self.model = model.to(device)
        self.data_root = './cifar_noise_data/'
        self.circle_length = circle_length
        self.model_name = model_name
        self.lr = 0.01
        self.noise_path = './cifar_noise_data/CIFAR-10_human.pt' if datasets == 'cifar10' else './cifar_noise_data/CIFAR-100_human.pt'
        self.trainset, self.testset, self.num_classes, self.num_training_samples = input_dataset(dataset=datasets,
                                                                                                 noise_type=noise_type,
                                                                                                 noise_path=self.noise_path,
                                                                                                 is_human=False,
                                                                                                 model_name=None)
        if datasets == 'cifar10':
            self.train_argumentation1 = CIFAR10('./cifar_noise_data/', transform=Augumentation_transforms1,
                                                download=False, noise_type=self.noise_type, noise_path=self.noise_path)
            self.train_augumentation2 = CIFAR10('./cifar_noise_data/', transform=Augumentation_transforms2,
                                                download=False, noise_type=self.noise_type, noise_path=self.noise_path)
        else:
            self.train_argumentation1 = CIFAR100('./cifar_noise_data/', transform=Augumentation_transforms1,
                                                 download=False, noise_type=self.noise_type, noise_path=self.noise_path)
            self.train_augumentation2 = CIFAR100('./cifar_noise_data/', transform=Augumentation_transforms2,
                                                 download=False, noise_type=self.noise_type, noise_path=self.noise_path)
        self.batch_size = batch_size
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=batch_size, shuffle=False, drop_last=True)
        self.test_loader = DataLoader(dataset=self.testset, batch_size=batch_size, shuffle=False, drop_last=True)
        self.noise_or_not = self.trainset.noise_or_not
        if forget_rate is None:
            self.forget_rate = self.trainset.actual_noise_rate
        else:
            self.forget_rate = forget_rate
        self.writer = Plot_graph(model_name=model_name, datasets=datasets, noise_type=noise_type)

    def evaluate(self, test_loader, model):
        model.eval()
        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).cuda()
            # print images.shape
            logits1 = model(images)
            outputs1 = F.log_softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()
        acc1 = 100 * float(correct1) / float(total1)
        model.train()
        return acc1

    def adjust_learning_rate(self, optimizer, epoch, max_epochs=250):
        if epoch < 0.25 * max_epochs:
            lr = 0.01
        elif epoch < 0.5 * max_epochs:
            lr = 0.005
        else:
            lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def first_stage(self, filter_mask=None):
        if filter_mask is not None:
            dataset = Mask_Select(self.trainset, filter_mask) + Mask_Select(self.train_argumentation1,
                                                                            filter_mask) + Mask_Select(
                self.train_augumentation2, filter_mask)
            train_loader_init = DataLoader(dataset=dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           shuffle=True, pin_memory=True)
        else:
            train_loader_init = DataLoader(dataset=self.trainset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           shuffle=True, pin_memory=True)
        desc = 'first_stage' if filter_mask is None else 'third stage'
        save_checkpoint = './state_dict/' + self.model_name + "_" + self.datasets + '_' + self.noise_type + '.pt'
        if filter_mask is not None:
            print("restore model from {}".format(save_checkpoint))
            self.model.load_state_dict(torch.load(save_checkpoint))

        ndata = self.trainset.__len__()
        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1, reduction='none').cuda()
        for epoch in range(1, self.n_epochs + 1):
            globals_loss = 0
            self.model.train()
            with torch.no_grad():
                accuracy = self.evaluate(self.test_loader, self.model)
            example_loss = np.zeros_like(self.noise_or_not, dtype=float)
            lr = self.adjust_learning_rate(optimizer=optimizer1, epoch=epoch, max_epochs=self.n_epochs)
            for i, (images, labels, indexes) in enumerate(tqdm(train_loader_init, desc=desc)):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                logits = self.model(images)
                loss_1 = criterion(logits, labels)

                for pi, cl in zip(indexes, loss_1):
                    example_loss += loss_1.sum().cpu().data.item()
                globals_loss += loss_1.sum().cpu().data.item()
                loss_1 = loss_1.mean()
                optimizer1.zero_grad()
                loss_1.backward()
                optimizer1.step()
            print(
                'epoch: {}, lr: {}, train_loss: {}, test_accuacy: {}'.format(epoch, lr, globals_loss / ndata, accuracy))
            # 保存文件

            if filter_mask is None:
                self.writer.first_stage_graph(epoch, globals_loss, accuracy, self.model_name)
                print("save_model " + save_checkpoint)
                torch.save(self.model.state_dict(), save_checkpoint)
            else:
                self.writer.third_stage_graph(epoch, globals_loss, accuracy, self.model_name)
                print("save_model " + save_checkpoint)
                torch.save(self.model.state_dict(), save_checkpoint)

    def second_stage(self, forget_rate):
        train_loader_detection = torch.utils.data.DataLoader(dataset=self.trainset,
                                                             batch_size=self.batch_size,
                                                             num_workers=0,
                                                             shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1, reduction='none').cuda()
        moving_loss_dic = np.zeros_like(self.noise_or_not)
        ndata = self.trainset.__len__()

        for epoch in range(1, self.max_epochs + 1):
            global_loss = 0
            self.model.train()
            with torch.no_grad():
                accuracy = self.evaluate(self.test_loader, self.model)
            example_loss = np.zeros_like(self.noise_or_not, dtype=float)
            t = (epoch % 10 + 1) / float(10)
            lr = (1 - t) * 0.01 + t * 0.001

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for i, (images, labels, indexes) in enumerate(tqdm(train_loader_detection, desc='second stage')):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()

                logits = self.model(images)
                loss_1 = criterion(logits, labels)

                for pi, cl in zip(indexes, loss_1):
                    example_loss[pi] = cl.cpu().data.item()

                globals_loss = loss_1.sum().cpu().data.item()
                loss_1 = loss_1.mean()
                optimizer.zero_grad()
                loss_1.backward()
                optimizer.step()
            example_loss = example_loss - example_loss.mean()
            moving_loss_dic = moving_loss_dic + example_loss
            ind_1_sorted = np.argsort(moving_loss_dic)
            loss_1_sorted = moving_loss_dic[ind_1_sorted]
            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(loss_1_sorted))

            noise_accuracy = np.sum(self.noise_or_not[ind_1_sorted[num_remember:]]) / float(
                len(loss_1_sorted) - num_remember)
            recall = np.sum(self.noise_or_not[ind_1_sorted[num_remember:]]) / float(np.sum(self.noise_or_not))
            mask = np.ones_like(self.noise_or_not, dtype=np.float32)
            self.writer.noise_acc(epoch=epoch, value=noise_accuracy, model_name=self.model_name)
            mask[ind_1_sorted[num_remember:]] = 0

            top_accuracy_rm = int(0.9 * len(loss_1_sorted))
            top_accuracy = 1 - np.sum(self.noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(
                len(loss_1_sorted) - top_accuracy_rm)
            print(
                'epoch:{}, lr: {}, train_loss: {}, test_acc: {}, noise_accuracy: {}, top 0.1 noise_accuracy: {}'.format(
                    epoch, lr, global_loss / ndata, accuracy, 1 - noise_accuracy, top_accuracy))
        return mask

    def detection(self):
        self.first_stage()
        base_forget_rate = self.forget_rate / self.circle_length
        for i in range(1, self.circle_length + 1):
            forget_rate = base_forget_rate * i
            filter_mask = self.second_stage(forget_rate=forget_rate)
            self.first_stage(filter_mask=filter_mask)
        self.writer.close()