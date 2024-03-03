from torch.utils.data import DataLoader
import torch
import torchvision
import os
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from torch import nn,load

BATCH_SIZE = 128
dir ='watermarking/Trigger/model/tiny'

def load_model(num,mode,dataset):
    if dataset == 'cifar10':
        model_base = '../cifar10/model'
    elif dataset == 'tiny':
        model_base='model/tiny'
    if mode == 'source_model':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        if dataset == 'cifar10':
            model.classifier[-1] = nn.Linear(in_feature, 10)
        elif dataset == 'tiny':
            model.classifier[-1] = nn.Linear(in_feature, 100)

        model.load_state_dict(load(os.path.join(model_base, "Source_Model", "source_model.pth")))
    elif mode == "extract_l":

        if 5<=num<10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny':
                model.fc = nn.Linear(in_feature, 100)
        elif num<5:
            model = torchvision.models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny':
                model.classifier[-1] = nn.Linear(in_feature, 100)

        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            if dataset == 'cifar10':
                model.classifier = nn.Linear(in_feature, 10)
            elif dataset == 'tiny':
                model.classifier = nn.Linear(in_feature, 100)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny':
                model.classifier[-1] = nn.Linear(in_feature, 100)

        model.load_state_dict(load(os.path.join(model_base,"Model_Extract_l", "extract_l_" + str(num) + ".pth")))
    return model

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
    model= load_model(iters, "extract_l", 'tiny')
    # model=load_model(iters, "extract_adv", 'tiny')
    model.train()
   # model=nn.DataParallel(model,device_ids=[0,1])

    model = model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr= 0.01,momentum=0.9, weight_decay=5e-4)#2.lr0.01 1e-3
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    # early_stopping = EarlyStopping(patience=8, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, verbose=1, patience=3)

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
    #
    #         if (i % 20 == 0):
    #             print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item(),"ASR:",1-acc)

        model.eval()
        correct=0
        for i, (x, y) in enumerate(val_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            predicted = torch.max(output, 1)[1].data.squeeze()
            correct += (predicted == b_y).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        # accu_best=val_accuracy
    print("Epoch:", epoch + 1, "accuracy:", val_accuracy)
        # if val_accuracy > accu_best:
#torch.save(model.state_dict(), os.path.join(dir, 'Model_Extract_adv',"adv_"  + str(iter) + ".pth"))
    torch.save(model.state_dict(), os.path.join(dir, 'Model_Extract_adv',"adv_"  + str(iter) + ".pth"))

    print(f"save model {iter}......")
    accu_best = val_accuracy
        # early_stopping(val_accuracy, model)
        # if early_stopping.early_stop:
        #     print("Early Stopping")
        #     break
        # scheduler.step(val_accuracy)
    return accu_best

if __name__ == "__main__":
    if os.path.exists(dir)==0:
        os.mkdir(dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    teacher = load_model(0, "source_model",'tiny')
    teacher = teacher.cuda()
    teacher.eval()
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

    accus = []
    start=0
    for iters  in range(start,20):
        accu = adv_train(iters,teacher,val_loader,train_loader)
        accus.append(accu)
    print(accus)
    #trigger:[0.565, 0.5754, 0.564, 0.5696, 0.567, 0.4886, 0.4636, 0.484, 0.4854, 0.4786, 0.5328, 0.526, 0.5276, 0.5352, 0.5178, 0.4584, 0.4444, 0.4622, 0.463, 0.4782]

