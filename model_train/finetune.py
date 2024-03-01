import torch
import torchvision
import os
from model_load import load_model

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
EPOCH = 100
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

def finetune_model(iter,teacher,cls,val_loader,train_loader):
    teacher = teacher.cuda()
    teacher.train()
    if cls == 'all':
        optimizer = torch.optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    elif cls == 'last':
        optimizer = torch.optim.SGD(teacher.classifier.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=3)
    accu_best = 0
    for epoch in range(EPOCH):
        teacher.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            teacher_output = teacher(b_x)
            loss = torch.nn.CrossEntropyLoss()(teacher_output, b_y)
            loss.backward()
            optimizer.step()

        ###############validation##############
        teacher.eval()
        correct=0
        for i, (x, y) in enumerate(val_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            output = teacher(b_x)
            _, predicted = output.max(1)
            correct += predicted.eq(y.cuda()).sum().item()
        val_accuracy=correct/len(val_loader.dataset)
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
    os.makedirs(os.path.join(dir, 'Finetune'), exist_ok=True)
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
