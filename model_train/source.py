import torch
import torchvision
import torch.optim as optim
import numpy as np
import os
from torch import nn

os.environ['CUDA_VISIBLE_DEVICES'] ='4'
# device_list=[int(i) for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]
dir = "../model/sac/"
os.makedirs(dir, exist_ok=True)
EPOCH=40
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss

def train_teacher_model(model,val_loader,train_loader):
    model = model.cuda()
    # model=nn.parallel.DataParallel(model,device_ids=device_list)

    optimizer = optim.SGD(model.parameters(), lr=1e-2,momentum=0.9, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=5)
    accu_best=0
    wait = 0
    for epoch in range(EPOCH):
        ###############train###################
        model.train()
        for i, (x,y) in enumerate(train_loader):
            x=x.type(torch.FloatTensor)
            y=y.long()
            b_x, _ = x.cuda(), y.cuda()
            optimizer.zero_grad()
            output = model(b_x)
            loss = torch.nn.CrossEntropyLoss()(output, y.cuda())
            loss.backward()
            optimizer.step()
    ###############validation###################
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
        # accu_best=val_accuracy
        print('Epoch:',epoch+1,f'val_accuracy:{val_accuracy},best:{accu_best}')
        if val_accuracy<accu_best+0.01:
            if wait==2:
                current_lr=optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = current_lr*0.1
                print(f'Can\'t wait to improve lr to {current_lr*0.1}')
                wait=0
                print(f'wait--->{wait}')
            else:
                wait+=1
                print(f'wait--->{wait}')
        else:
            wait=0
            print(f'wait--->{wait}')
        if val_accuracy>accu_best:
            torch.save(model.state_dict(), os.path.join(dir,'Source_Model','source_model.pth'))
            print("save model")
            accu_best=val_accuracy
        early_stopping(val_accuracy, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        scheduler.step(val_accuracy)
    return accu_best

if __name__ == "__main__":
    os.makedirs(os.path.join(dir, 'Source_Model'), exist_ok=True)
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torchvision.transforms.Normalize(mean=[0.4804, 0.4482, 0.3976], std=[0.2770, 0.2691, 0.2822])
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torchvision.transforms.Normalize(mean=[0.4804, 0.4482, 0.3976], std=[0.2770, 0.2691, 0.2822])

    ])
    val_data = torchvision.datasets.ImageFolder(root='../data/val__/', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
    train_data = torchvision.datasets.ImageFolder(root='../data/extend_/', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    teacher = torchvision.models.vgg16_bn(pretrained=True)
    in_feature = teacher.classifier[-1].in_features
    teacher.classifier[-1] = torch.nn.Linear(in_feature, 100)

    print("Begin training model:",  "source model:vgg")

    accu = train_teacher_model(teacher,val_loader,train_loader )
    print(accu)#0.6226  #0.7434
