import torch
import torchvision
import os
import torch.nn.functional as F
from model_load import load_model

os.environ['CUDA_VISIBLE_DEVICES'] ='3'
EPOCH = 100
BATCH_SIZE=128
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

def reset(cls):
    if cls == 'resnet':
        model = torchvision.models.resnet18(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
    elif cls == 'vgg':
        model = torchvision.models.vgg13(pretrained=True)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 100)

    elif cls == 'dense':
        model = torchvision.models.densenet121(pretrained=True)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 100)

    elif cls == 'mobile':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 100)
    model.cuda()
    return model

def train_student_model(iter,teacher,cls,val_loader,train_loader):
    model = reset(cls)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.1,momentum=0.9, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=5)
    alpha = 0.9
    T = 20
    accu_best = 0
    for epoch in range(EPOCH):
        ###############train###################
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            optimizer.zero_grad()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            output = model(b_x)
            loss = torch.nn.KLDivLoss()(F.log_softmax(output / T, dim=1),F.softmax(teacher_output / T, dim=1)) * (alpha * T * T)+ F.cross_entropy(output, pred) * (1. - alpha)
            loss.backward()
            optimizer.step()

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
        print('Epoch:',epoch+1,f'val_accuracy:{val_accuracy},best:{accu_best}')
        # if val_accuracy<accu_best+0.01:
        #     if wait==2:
        #         current_lr=optimizer.param_groups[0]['lr']
        #         optimizer.param_groups[0]['lr'] = current_lr*0.1
        #         print(f'Decrease lr to {current_lr*0.1}')
        #         wait=0
        #         print(f'wait ---> {wait}')
        #     else:
        #         wait+=1
        #         print(f'wait--->{wait}')
        # else:
        #     wait=0
        #     print(f'wait--->{wait}')
        if val_accuracy > accu_best:
            torch.save(model.state_dict(), os.path.join(dir, 'Model_Extract_p',"extract_p_" + str(iter) + ".pth"))
            print("save model......")
            accu_best = val_accuracy
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        scheduler.step(val_accuracy)
    return accu_best

if __name__ == "__main__":
    os.makedirs(os.path.join(dir, 'Model_Extract_p'), exist_ok=True)
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
        if iters < 5:
            cls = 'vgg'
        elif 5<=iters<10:
            cls = 'resnet'
        elif 10 <= iters < 15:
            cls = 'dense'
        elif 15 <= iters:
            cls = 'mobile'
        print("Begin training model:", iters,"student model:",cls)
        accu = train_student_model(iters,teacher,cls,val_loader,train_loader)
        accus.append(accu)
        print("Model {} has been trained and the accuracy is {}".format(iters, accu))

    print(accus)
#[0.5932, 0.602, 0.5936,# 0.591, 0.6012, 0.5608, 0.5536, 0.5384, 0.563, 0.5534, 0.6038, 0.6028, 0.606, 0.603, 0.591, 0.5532, 0.557, 0.5616, 0.561, 0.555]
#trigger:[0.6256, 0.6272, 0.6218, 0.625, 0.6162, 0.5836, 0.5846, 0.5794, 0.5762, 0.5814, 0.6376, 0.6426, 0.6382, 0.6396, 0.6456, 0.5766, 0.5712, 0.5742, 0.5682, 0.5802]