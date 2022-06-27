# -*- coding: utf-8 -*-
# @time: 2022/5/16 17:12
# @Author: 坤
# @file: dataloader.py
"""
这个文件我写了一些接口用于训练，在使用的时候可能需要进行调试
在调试的时候，需要注意，文件的路径需要修改，参数需要修改

"""
import os.path
import time
from torch.autograd import Variable
from cifar_noise_data.datasets import input_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm, trange
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 这是获取data_loader的接口，创建数据，这里要注意的是，你在创建Cifar_dataloader对象的时候，需要指定你的数据集dataset
# noisy_type 一共有以下值
class Cifar_dataloader():
    def __init__(self,batch_size=64,dataset='cifar10',noise_type='worst',model_name=None,is_human=False):
        """
        noise_type: 指定数据的噪点类型,共有 clearn, worst, worst_label, aggre, rand1, rand2, rand3, clean100, noisy100这些选项,并且如果dataset是100cifar的话,必须选择clean100 和 noisy100
        """
        self.dataset =dataset
        self.model_name = model_name
        noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
        self.noise_type = noise_type_map[noise_type]
        self.is_human = is_human
        self.batch_size = batch_size
        self.noise_path = r'./cifar_noise_data/CIFAR-10_human.pt' if dataset == 'cifar10' \
            else r'./cifar_noise_data/CIFAR-100_human.pt'

    # 获取数据的函数,用于获取train_loader,test_loader,num_classes 
    def get_dataloader(self,num_workers):
        train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
            dataset=self.dataset,
            noise_type=self.noise_type,
            is_human=self.is_human,
            noise_path=self.noise_path,
            model_name=self.model_name,
        )
        train_loader = DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
        return train_loader,test_loader,num_classes



# 这个类是用于绘图的类，用的是tensorboard
# 这一段可以改成visdom
class Plot_graph():
    def __init__(self,model_name):
        datatime = time.strftime('%Y%m%d%H%M')
        path = model_name + datatime
        self.writer = SummaryWriter('./runs/'+path+'/')

    def loss_graph(self,epoch,value,model_name):
        self.writer.add_scalar('{}/Loss'.format(model_name),value,epoch)

    def acc_graph(self,epoch,train_acc,test_acc,model_name):
        self.writer.add_scalar('{}/acc/train'.format(model_name),train_acc,epoch)
        self.writer.add_scalar('{}/acc/test'.format(model_name),test_acc,epoch)

    def close(self):
        self.writer.close()

# 损失函数接口
def loss_cross_entropy(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    ## Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch

# 训练器接口，用于把数据和模型以及优化器为参数塞进去，调用函数就能进行训练
# 这个是训练器,作用是封装训练时的步骤，做到直接创建对象，塞进参数就能调用run进行训练,并且在训练的时候能够做到用tensorboard完成绘图,训练完后自动保存模型
class Trainer():
    def __init__(self,optimizer,loss_function,n_epoch,model,dataset,noise_type,batch_size,model_name,num_workers):
        self.model = model
        self.trainloader,self.testloader,self.num_classes = Cifar_dataloader(dataset=dataset,
                                                            batch_size=batch_size,
                                                            noise_type=noise_type,model_name=model_name,
                                                            ).get_dataloader(num_workers=num_workers)
        self.n_epoch = n_epoch

        self.loss_function = loss_function
        self.writer = Plot_graph(model_name=model_name)
        self.optimizer = optimizer
        self.model.to(device)
        self.train_acc = []
        self.test_acc = []
        self.losses = []
        self.model_name = model_name

    def accuracy(self,logit, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        output = F.softmax(logit, dim=1)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def train(self,epoch):
        self.model.train()
        train_total = 0
        train_correct = 0
        total_loss = 0
        for i, (images, labels, indexes) in enumerate(tqdm(self.trainloader)):
            ind = indexes.cpu().numpy().transpose()
            batch_size = len(ind)

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            # Forward + Backward + Optimize
            logits = self.model(images)

            prec,_ = self.accuracy(logits, labels, topk=(1, 5))
            # prec = 0.0
            train_total += 1
            train_correct += prec
            loss = F.cross_entropy(logits, labels, reduce=True)
            self.losses.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.writer.loss_graph(epoch,loss)
            total_loss += loss.data
        train_acc = float(train_correct) / float(train_total)
        return train_acc,total_loss

    def save(self,model_name):
        torch.save(self.model.state_dict(),'./state_dict/'+model_name+'.pth')

    # 调用run函数,完成模型的训练和评估,以及绘图和参数的保存
    def run(self):
        for epoch in range(self.n_epoch):
            train_acc,total_loss = self.train(epoch)
            test_acc = self.evaluate(epoch)
            # self.train_acc.append(train_acc)
            # self.test_acc.append(test_acc)
            self.save(self.model_name)
            print("Epoch: {}, test_acc: {}, train_acc: {}, loss: {}".format(epoch+1,test_acc,train_acc,total_loss))
            self.writer.loss_graph(epoch,total_loss,self.model_name)
            self.writer.acc_graph(epoch, train_acc, test_acc,self.model_name)
        self.writer.close()

    # 用于评估模型的函数, 是在test数据集上运行评估目前模型的结果
    def evaluate(self,epoch):
        self.model.eval()  # Change model to 'eval' mode.
        # print('previous_best', best_acc_)
        correct = 0
        total = 0
        for images, labels, _ in self.testloader:
            images = Variable(images).cuda()
            logits = self.model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
        acc = 100 * float(correct) / float(total)
        return acc


# 需要注意的是,在运行程序时需要做到在终端上运行,否则会报路径方面的错误
# 下面的函数是用于解析终端指令而存在的
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


if __name__ == "__main__":
    trainloader,testloader,num_classes = Cifar_dataloader().get_dataloader()
    for (imgs,targets,index) in trainloader:
       print(index)
       break

