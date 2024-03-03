from torch.utils.data import DataLoader
import torch
import torchvision
import os
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] ='6'
device_list=[int(i) for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

dir = "../model/sac/"
os.makedirs(dir, exist_ok=True)
EPOCH=40
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
    optimizer = optim.SGD(model.parameters(), lr= 0.1,momentum=0.9, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=5)
    alpha = 0.9
    T = 20
    accu_best = 0
    for epoch in range(EPOCH):
    #     ###############train###################
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            optimizer.zero_grad()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            output = model(b_x)
            loss = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),F.softmax(teacher_output / T, dim=1)) * (alpha * T * T)+ F.cross_entropy(output, pred) * (1. - alpha)
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
        if val_accuracy > accu_best:
            torch.save(model.state_dict(), os.path.join(dir, 'Model_Extract_p',"extract_p_" + str(iter) + ".pth"))
            print("save model......")
            accu_best = val_accuracy
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        if optimizer.param_groups[0]['lr']<0.0001:
            print("Early Stopping")
            break
        scheduler.step(val_accuracy)
    return accu_best

if __name__ == "__main__":

    os.makedirs(os.path.join(dir, 'Model_Extract_p'),exist_ok=True)

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
    train_data = torchvision.datasets.ImageFolder(root='../data/extend_', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    teacher = torchvision.models.vgg16_bn(pretrained=False)
    in_feature = teacher.classifier[-1].in_features
    teacher.classifier[-1] = torch.nn.Linear(in_feature, 100)
    teacher.load_state_dict(torch.load(dir + "/Source_Model/source_model.pth"))
    teacher = teacher.eval().cuda()
    accus = []

    start=10
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

    print(accus)#[0.5932, 0.602, 0.5936,# 0.591, 0.6012, 0.5608, 0.5536, 0.5384, 0.563, 0.5534, 0.6038, 0.6028, 0.606, 0.603, 0.591, 0.5532, 0.557, 0.5616, 0.561, 0.555]
# ori:[0.5644, 0.564, 0.5712, 0.5108, 0.4894, 0.4956, 0.5032, 0.5048]
#trigger:[0.6256, 0.6272, 0.6218, 0.625, 0.6162, 0.5836, 0.5846, 0.5794, 0.5762, 0.5814, 0.6376, 0.6426, 0.6382, 0.6396, 0.6456, 0.5766, 0.5712, 0.5742, 0.5682, 0.5802]

#extend_renew32batchsize: 0.6218 0.5854 0.6098 0.5442 0.6168 0.6196 0.6106 0.6258 0.5912 0.6068