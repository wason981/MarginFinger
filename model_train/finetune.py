#import cv2
from torch.utils.data import DataLoader
import torch
import torchvision
import os
import torch.optim as optim
from torch import nn
import numpy as np
BATCH_SIZE = 256
EPOCH = 30
dir ='watermarking/Trigger/model/tiny'
pretrain=0
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, trace_func=print):
        self.patience = patience
        self.best_accuracy = None
        self.counter = 0
        self.trace_func = trace_func
        self.early_stop = False
    def __call__(self, accuracy):
        if self.best_accuracy is None:
            self.best_accuracy = accuracy
        elif accuracy < self.best_accuracy :
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_accuracy = accuracy
            self.counter = 0

def finetune_model(iter,teacher,cls,val_loader,train_loader):
    teacher = teacher.cuda()
    teacher.train()
    if cls == 'all':
        optimizer = optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    elif cls == 'last':
        optimizer = optim.SGD(teacher.classifier.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    early_stopping=EarlyStopping(patience=5, verbose=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=3)
    accu_best = 0
    for epoch in range(EPOCH):
        teacher.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            loss = nn.CrossEntropyLoss()(teacher_output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        teacher.eval()
        num = 0
        total_num = 0
        for i, (x, y) in enumerate(val_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = teacher(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]
        val_accuracy = num / total_num
        print('Epoch:',epoch+1,f'val_accuracy:{val_accuracy},best:{accu_best}')

        if val_accuracy > accu_best:
            torch.save(teacher.state_dict(), os.path.join(dir,'Finetune', "finetune" + str(iter) + ".pth"))
            print("save model......")
            accu_best = val_accuracy
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        scheduler.step(val_accuracy)
    return accu_best

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if os.path.exists(dir) == 0:
        os.mkdir(dir)
    teacher = torchvision.models.vgg16_bn(pretrained=False)
    in_feature = teacher.classifier[-1].in_features
    teacher.classifier[-1] = torch.nn.Linear(in_feature, 100)
    teacher.load_state_dict(torch.load(dir+"Source_Model/source_model.pth"))
    teacher=teacher.eval().cuda()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_data = torchvision.datasets.ImageFolder(root='data/val__/', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    train_data = torchvision.datasets.ImageFolder(root='data/attacker_/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    start=0
    accus = []
    for iter in range(start,20):
        iters = iter
        if iters<10:
            cls = 'all'
        elif 10<=iters<20:
            cls = 'last'
        print("Begin training model:", iters,"finetune model:",cls)
        accu = finetune_model(iters,teacher,cls,val_loader,train_loader)
        accus.append(accu)
        print("Model {} has been trained and the accuracy is {}".format(iters, accu))
    print(accus)
# [0.7398, 0.7422, 0.7392, 0.7382, 0.7424, 0.7338, 0.7306, 0.73, 0.7296, 0.7218, 0.7258, 0.7272, 0.7242, 0.725, 0.7258, 0.725, 0.7242, 0.7268, 0.7236, 0.724]
