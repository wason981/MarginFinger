import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
dir ='../model/tiny'

BATCH_SIZE = 256
LR = 0.001
prune_ratio = 0.90

def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx


def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0


class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.output = output

    def close(self):
        self.hook.remove()

def find_smallest_neuron(hook_list,prune_list):
    activation_list = []
    for j in range(len(hook_list)):
        activation = hook_list[j].output
        for i in range(activation.shape[1]):
            activation_channel = torch.mean(torch.abs(activation[:,i,:,:]))
            activation_list.append(activation_channel)

    activation_list1 = []
    activation_list2 = []

    for n, data in enumerate(activation_list):
        if n in prune_list:
            pass
        else:
            activation_list1.append(n)
            activation_list2.append(data)

    activation_list2 = torch.tensor(activation_list2)
    prune_num = torch.argmin(activation_list2)
    prune_idx = activation_list1[prune_num]

    return prune_idx

def finetune_step(model, dataloader, criterion):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for i,(inputs,labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels=labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1)*inputs.shape[0]>= 2056:
            break


def value(model, dataloader):
    model.eval()
    num = 0
    total_num = 0
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()
        num += (pred == b_y).sum().item()
        total_num += pred.shape[0]

    accu = num / total_num
    return accu

def run_model(model, dataloader):
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()


def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx


def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0


def plot_figure(mem,length):
    plt.figure(1)
    acc = np.squeeze(mem)
    plt.plot(
        np.squeeze(np.array(acc)[:, 0])/length,
        np.squeeze(np.array(acc)[:, 1]),
        'b',
        label='Clean Classification Accuracy'
    )
    plt.xlabel("Ratio of Neurons Pruned")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig("./rate_pruned.jpg")


def fine_pruning(model, train_loader, test_loader):
    model = model.cuda()
    module_list = []###存放要剪枝的模块
    neuron_num = []###要剪枝的模块输出神经元个数
    hook_list = []#存放要剪枝的模块的hook
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_list.append(module)
            neuron_num.append(module.out_channels)
            hook_list.append(FeatureHook(module))

    neuron_num = np.array(neuron_num)
    max_id = np.sum(neuron_num)##要剪枝的模块的输出神经元个数总和

    neuron_list = []
    mask_list = []
    for i in range(neuron_num.shape[0]):
        neurons = list(range(neuron_num[i]))#神经元序号
        neuron_list.append(neurons)
        prune_filter = prune.identity(module_list[i], 'weight')
        mask_list.append(prune_filter)

    prune_list = []
    init_val = value(model, test_loader)###初始模型的准确率
    acc = []
    length = deepcopy(len(neuron_list))##模块个数
    total_length = 0###神经元个数总和
    for i in range(length):
        total_length += len(neuron_list[i])
    print("Total number of neurons is",total_length)
    for i in range(int(np.floor(0.8*total_length))):###80%的神经元：3379
        if i % 20 == 0:
            run_model(model, train_loader)
        idx = find_smallest_neuron(hook_list, prune_list)
        prune_list.append(idx)
        prune_neuron(mask_list, idx, neuron_num)
        if i % 50 == 0:
            finetune_step(model, train_loader, criterion=torch.nn.CrossEntropyLoss())
        if i % 50 == 0:
            new_val = value(model, test_loader)
            print("neuron remove:", i, "init_value:", init_val, "new_value:", new_val)
            acc.append([i, new_val])

        if (
            np.floor(20*i/total_length)-np.floor(20*(i-1)/total_length)
        ) == 1:###4224分20个阶梯，如果进了一个阶梯就保存一个模型
            iter = int(np.floor(20*i/total_length))
            torch.save(model, os.path.join(dir,'Fine-Pruning',"prune_model_" + str(iter)+".pth"))
            print("Saving model! Model number is:",iter)

    mem = np.array([acc])
    return mem,length


if __name__ == "__main__":
    if os.path.exists(dir) == 0:
        os.mkdir(dir)
        print("Making directory!")

    teacher = torchvision.models.vgg16_bn(weights=False)
    in_feature = teacher.classifier[-1].in_features
    teacher.classifier[-1] = torch.nn.Linear(in_feature, 100)
    teacher.load_state_dict(torch.load(dir+"/Source_Model/source_model.pth"))
    teacher.eval()
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.ImageFolder(root='data/val__/', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    train_data = torchvision.datasets.ImageFolder(root='data/attacker_/', transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    mem, length = fine_pruning(teacher, train_loader, test_loader)
    plot_figure(mem, length)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    # train_data = dataset('dataset_common.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)



    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=BATCH_SIZE, shuffle=False)
    # train_data = dataset('dataset_attack_1.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)


