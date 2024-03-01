from torchvision import  models
from torch import nn,load
import os
from query_attack import DetectionAE

def load_model(num,mode,dataset,path=None):
    if path is not None:
        model_base=path
    else:
        if dataset=='cifar10':
            model_base = '../model/cifar10'
        else:
            model_base='../model/tiny_imagenet'
    #源模型
    if mode == 'source_model':
        model = models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        if dataset=='cifar10':
            model.classifier[-1] = nn.Linear(in_feature, 10)
        elif dataset=='tiny_imagenet':
            model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Source_Model", "source_model.pth")))

    #模型提取-概率
    elif mode == 'extract_p':
        if num<5:
            model = models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        elif 5<=num<10:
            model = models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.fc = nn.Linear(in_feature, 100)
        elif 15>num>=10:
            model = models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            if dataset == 'cifar10':
                model.classifier = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier = nn.Linear(in_feature, 100)
        elif 20>num>=15:
            model = models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Model_Extract_p", "extract_p_" + str(num) + ".pth")))

    #模型提取-标签
    elif mode == "extract_l":
        if num<5:
            model = models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        if 5<=num<10:
            model = models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.fc = nn.Linear(in_feature, 100)
        elif 15>num>=10:
            model = models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            if dataset == 'cifar10':
                model.classifier = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier = nn.Linear(in_feature, 100)
        elif 20>num>=15:
            model = models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Model_Extract_l", "extract_l_" + str(num) + ".pth")))

    elif mode == "teacher_kd":
        if num>=3:
            model = models.resnet34(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.fc = nn.Linear(in_feature, 100)
        else:
            model = models.vgg19_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"model", "student_model_tk_" + str(num) + ".pth")))

    #无关模型
    elif mode == "irrelevant":
        if num < 5:
            model = models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        elif 10>num>=5:
            model = models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.fc = nn.Linear(in_feature, 100)
        elif 15 > num >= 10:
            model = models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            if dataset == 'cifar10':
                model.classifier = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier = nn.Linear(in_feature, 100)
        elif 20 > num >= 15:
            model = models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Irrelevant", "irrelevant_" + str(num) + ".pth")))

    #迁移学习
    elif mode == 'transfer':
        model =  models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        if dataset == 'cifar10':
            model.classifier[-1] = nn.Linear(in_feature, 10)
        elif dataset == 'tiny_imagenet':
            model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Transfer_Learning", "transfer_" + str(num) + ".pth")))

    # 迁移学习-无关
    elif mode == 'transfer_irrelevant':
        if num < 5:
            model =  models.vgg16_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        elif num >= 5:
            model =  models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.fc = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Transfer_Learning", "irrelevant_" + str(num) + ".pth")))

    # 模型提取-对抗
    elif mode == 'extract_adv':
        if num < 5:
            model = models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        elif 5 <= num < 10:
            model = models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            if dataset == 'cifar10':
                model.fc = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.fc = nn.Linear(in_feature, 100)
        elif 15 > num >= 10:
            model = models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            if dataset == 'cifar10':
                model.classifier = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier = nn.Linear(in_feature, 100)
        elif 20 > num >= 15:
            model = models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Model_Extract_adv", "adv_" + str(num) + ".pth")))

    #剪枝
    elif mode == "fine-pruning":
        model = load(os.path.join(model_base,"Fine-Pruning","prune_model_"+str(num)+".pth"))

    ##微调
    elif mode == 'finetune':
        model = models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        if dataset == 'cifar10':
            model.classifier[-1] = nn.Linear(in_feature, 10)
        elif dataset == 'tiny_imagenet':
            model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"Finetune", "finetune" + str(num) + ".pth")))

    elif mode == 'surrogate':
        model = models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        if dataset == 'cifar10':
            model.classifier[-1] = nn.Linear(in_feature, 10)
        elif dataset == 'tiny_imagenet':
            model.classifier[-1] = nn.Linear(in_feature, 100)
        model.load_state_dict(load(os.path.join(model_base,"surrogate", "surrogate_" + str(num) + ".pth")))

    elif mode == "query_attack":
        model = DetectionAE()
        model.load_state_dict(load(f"./fingerprint/query_attack/autoencoder/model_best.pth"))
    return model