import torch
import torchvision
import os
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from model_load import load_model
os.environ['CUDA_VISIBLE_DEVICES'] ='3'
BATCH_SIZE = 128
dir='../model/tiny_imagenet'

class EarlyStopping:
    def __init__(self, patience=7, delta=0, trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_acc):
        self.cur_acc = val_acc
        if self.best_acc is None:
            self.best_acc = self.cur_acc
        elif self.cur_acc < self.best_acc + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = self.cur_acc
            self.counter = 0

def denormalize(image):
    image_data = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    image_data = np.array(image_data)

    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:,i,:,:] = image[:,i,:,:]*image_data[1,i] + image_data[0,i]

    return img_copy

def normalize(image):
    image_data = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:, i, :, :] = (image[:, i, :, :] - image_data[0, i])/image_data[1,i]

    return img_copy

def PGD(model,image,label):
    label = label.cuda()
    loss_func1 = torch.nn.CrossEntropyLoss()
    image_de = denormalize(deepcopy(image))#回归到0到1
    image_attack = deepcopy(image)
    image_attack = image_attack.cuda()
    image_attack = Variable(image_attack, requires_grad=True)
    alpha = 1/256
    epsilon = 4/256

    for iter in range(30):
        image_attack = Variable(image_attack, requires_grad=True)
        output = model(image_attack)
        loss = -loss_func1(output,label)
        loss.backward()
        grad = image_attack.grad.detach().sign()
        image_attack = image_attack.detach()
        image_attack = denormalize(image_attack)
        image_attack -= alpha*grad
        eta = torch.clamp(image_attack-image_de,min=-epsilon,max=epsilon)
        image_attack = torch.clamp(image_de+eta,min=0,max=1)
        image_attack = normalize(image_attack)
    pred_prob = output.detach()
    pred = torch.argmax(pred_prob, dim=-1)
    acc_num = torch.sum(label == pred)
    num = label.shape[0]
    acc = acc_num/num
    acc = acc.data.item()

    return image_attack.detach(), acc

def adv_train(iter,teacher,val_loader,train_loader):
    teacher.eval()
    model= load_model(iter, "extract_l", 'tiny_imagenet')
    model.train().cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    accu_best = 0
    for epoch in range(2):
        ###############train###################
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            x_adv,acc = PGD(model,b_x,pred)#l模型 训练集 教师标签
            output = model(b_x)
            output_adv = model(x_adv)
            loss = loss_func(output, pred) + loss_func(output_adv, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 20 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item(),"ASR:",1-acc)

        ###############validation##############
        model.eval()
        correct=0
        for i, (x, y) in enumerate(val_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            output = model(b_x)
            _, predicted = output.max(1)
            correct += predicted.eq(y.cuda()).sum().item()
        val_accuracy=correct/len(val_loader.dataset)
        print('Epoch:', epoch + 1, f'val_accuracy:{val_accuracy},best:{accu_best}')
    torch.save(model.state_dict(), os.path.join(dir, 'Model_Extract_adv',"adv_"  + str(iter) + ".pth"))
    print(f"save model {iter}......")
    accu_best = val_accuracy
    return accu_best

if __name__ == "__main__":
    os.makedirs(os.path.join(dir, 'Model_Extract_adv'), exist_ok=True)
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = torchvision.datasets.ImageFolder(root='../data/tiny_imagenet/attacker/front_100', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = torchvision.datasets.ImageFolder(root='../data/tiny_imagenet/val/front_100', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    teacher = load_model(0, "source_model",'tiny_imagenet')
    teacher = teacher.eval().cuda()
    accus = []

    start=0
    for iter in range(start,20):
        accu = adv_train(iter,teacher,val_loader,train_loader)
        accus.append(accu)
    print(accus)
#trigger:[0.565, 0.5754, 0.564, 0.5696, 0.567, 0.4886, 0.4636, 0.484, 0.4854, 0.4786, 0.5328, 0.526, 0.5276, 0.5352, 0.5178, 0.4584, 0.4444, 0.4622, 0.463, 0.4782]

