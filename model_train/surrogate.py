from torch.utils.data import DataLoader
import torch
import torchvision
import os
import torch.optim as optim
from torch import nn
BATCH_SIZE = 256
EPOCH = 300
dir ='../model/tiny'

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

def train_student_model(iter, teacher,):
    model = torchvision.models.vgg16_bn(pretrained=True)
    in_feature = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_feature, 100)
    model=model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=5)
    wait = 0
    count = 0
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
            loss = nn.CrossEntropyLoss()(output, pred.cuda())
            loss.backward()
            optimizer.step()
        ###############validation###################
        model.eval()
        num = 0
        total_num = 0
        for i, (x, y) in enumerate(val_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]
        val_accuracy = num / total_num
            # accu_best=val_accuracy
        print('Epoch:',epoch+1,f'val_accuracy:{val_accuracy},best:{accu_best}')
        if val_accuracy<accu_best+0.01:
            if wait==3:
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
            torch.save(model.state_dict(), os.path.join(dir, 'surrogate',"surrogate_" + str(iter) + ".pth"))
            print("save model......")
            accu_best = val_accuracy
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        scheduler.step(val_accuracy)
    return accu_best


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    if os.path.exists(dir) == 0:
        os.mkdir(os.path.join(dir, 'surrogate'))
    teacher = torchvision.models.vgg16_bn(pretrained=False)
    in_feature = teacher.classifier[-1].in_features
    teacher.classifier[-1] = torch.nn.Linear(in_feature, 100)
    teacher.load_state_dict(torch.load(dir + "/Source_Model/source_model.pth"))
    teacher = teacher.eval().cuda()
    accus = []
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_data = torchvision.datasets.ImageFolder(root='../data/val__/', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    train_data = torchvision.datasets.ImageFolder(root='../data/attacker_/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    iter=2
    # for iter in range(3):
    print("Beigin training model:", iter, "student model:", "vgg")
    accu = train_student_model(iter, teacher)
    print(
        "Model {} has been trained and the accuracy is {}".format(
            iter, accu
        )
    )
## 0.6732 0.648 0.6844
    # nohup python surrogate.py --gpu 4 --model_index 0 &
    # nohup python surrogate.py --gpu 5 --model_index 1 &
    # python surrogate.py --gpu 6 --model_index 2
