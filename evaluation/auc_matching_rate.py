import argparse
import os
from model_load import load_model
import numpy as np
import torch
import xlsxwriter as xw
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch import argmax
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader
import utils
from torch.utils.data import Dataset, DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

BATCH_SIZE = 100
img_class=0

class CustomDataset(Dataset):
    def __init__(self, data_dict_list):
        self.data = data_dict_list['data']
        self.label = data_dict_list['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx])
class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
    def close(self):
        self.hook.remove()


def calculate_auc(list_a, list_b):
    l1,l2 = len(list_a),len(list_b)
    y_true,y_score = [],[]
    for i in range(l1):
        y_true.append(1)
    for i in range(l2):
        y_true.append(0)###真实标签[0,0,0,0,0,1,1,1,1,1]
    y_score.extend(list_a)
    y_score.extend(list_b)###预测分数(概率）[0.09,0.08,0.08,... 0.99,0.98,0.97]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)###设置不同的阈值，根据预测分数与不同阈值画出roc
    # auc_score = roc_auc_score(y_true, y_score)
    auc__=auc(fpr, tpr)
    return auc__

def test_similarity(models,test_loader):
    results=[]
    similarity=[]
    for i in range(len(models)):
        model = models[i]
        model = model.cuda()
        model.eval()
        if i==0:
            result,mean,std=cal_pred(model,test_loader,i)
        else:
            result,_,_=cal_pred(model,test_loader,i)
        results.append(result)
        model=model.cpu()
    for i in range(1,len(results)):
        simi=np.sum(results[0]==results[i])/result.shape[0]
        similarity.append(simi)
    return similarity,mean,std

def cal_pred(model,dataloader,it):
    result=[]
    distance = torch.tensor([]).cuda()
    # std = 0
    for i,(x,y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        # y=torch.full((y.shape[0],),3).cuda()####重新设置猫类别为3
        model=model.cuda()
        output = model(x)
        output = output.detach()
        pred = argmax(output,dim=-1)
        result.append(pred.data.cpu().numpy())
        if it==0:

            y_pred = torch.softmax((output), dim=1)
            y_pred_clone = y_pred.clone()
            #######label#####
            right_pred = y_pred[torch.arange(len(y_pred)), i]
            y_pred_clone[torch.arange(len(y_pred)), i] = -1000
            second_pred = torch.max(y_pred_clone, axis=1).values
            #####no label#####
            # right_pred = torch.max(y_pred, axis=1).values
            # y_pred_clone[torch.arange(len(y_pred)), torch.argmax(y_pred, axis=1)] = -100
            # second_pred = torch.max(y_pred_clone, axis=1).values
            #################
            distance=torch.cat((distance, right_pred - second_pred))
    result=np.concatenate(result)
    mean = distance.mean()
    std = distance.std()
    return result,mean,std

def auc_result(args,models_list,worksheet1=None,row=None):
    if args.dataset=='cifar10':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif args.dataset=='tiny':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # for it in range(940, 960, 10):
    for it in range(50, 760,10):
    # for it in range(50, len(os.listdir()), 1000):
        iter_path=os.path.join(args.img_path,f'{it}')
        try:
            dataset=ImageFolder(os.path.join(iter_path),transform=transform)
        except:
            print(iter_path,'not exist')
        test_loader = data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            drop_last=False,
            num_workers=0,
            shuffle=False
        )
        with  torch.no_grad():
            similarity,mean,std = test_similarity(models_list, test_loader)

        list1 = similarity[:20]  # ext-p
        list2 = similarity[20:40]  # ext-l
        list3 = similarity[40:60]  # ext-adv
        list4 = similarity[60:65]  # prune
        list5 = similarity[65:85]  # finetune
        list6 = similarity[85:95]  # cifar10c ft
        list7 = similarity[95:105]  # cifar10c irrelevant
        list8 = similarity[105:125]  # irrelevant

        auc_p = calculate_auc(list1, list8)  ####extract-p
        auc_l = calculate_auc(list2, list8)  ####extract-l
        auc_adv = calculate_auc(list3, list8)  ####extract-adv
        auc_prune = calculate_auc(list4, list8)  ####prune
        auc_finetune = calculate_auc(list5, list8)  ###finetune-a
        auc_10C = calculate_auc(list6, list7)  ####transfer-10C
        auc_list = [round(mean.item(), 2), round(std.item(), 2),
                    round(auc_p, 2), round(auc_l, 2), round(auc_adv, 2),
                    round(auc_prune, 2), round(auc_finetune, 2),round(auc_10C, 2)
                    ]

        worksheet1.write_row('A' + str(row), auc_list)
        row += 1
        for j in auc_list:
            sys.stdout.write(str("{:.2f}".format(j)))
            sys.stdout.write(str(' '))
        sys.stdout.write(str('\n'))
def auc_pickle(ip_erase):
    data_set = utils.load_result(
        f"./fingerprint/{ip_erase}/margin_fingerprint.pkl"
    )
    data_set['data']=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(data_set['data'])
    dataset = CustomDataset(data_set)
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    with  torch.no_grad():
        similarity = test_similarity(models_list, dataloader)

    list1 = similarity[:20]  # ext-p
    list2 = similarity[20:40]  # ext-l
    list3 = similarity[40:60]  # ext-adv
    list4 = similarity[60:65]  # prune
    list5 = similarity[65:85]  # finetune
    list6 = similarity[85:95]  # cifar10c ft
    list7 = similarity[95:105]  # cifar10c irrelevant
    list8 = similarity[105:125]  # irrelevant

    auc_p = calculate_auc(list1, list8)  ####extract-p
    auc_l = calculate_auc(list2, list8)  ####extract-l
    auc_adv = calculate_auc(list3, list8)  ####extract-adv
    auc_prune = calculate_auc(list4, list8)  ####prune
    auc_finetune = calculate_auc(list5, list8)  ###finetune-a
    auc_10C = calculate_auc(list6, list7)  ####transfer-10C
    # auc_list = [round(auc_p, 2), round(auc_l, 2), round(auc_adv, 2),
    #             round(auc_prune, 2), round(auc_finetune, 2),round(auc_10C, 2)]
    print(
        "pro:",
        auc_p,
        "lab:",
        auc_l,
        "adv:",
        auc_adv,
        "fp:",
        auc_prune,
        "ft:",
        auc_finetune,
        "cif:",
        auc_10C,
    )

import sys
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="AUC Calculation")
    parser.add_argument(
        "--root_path",
        type=str,
        # default='/data/huangcb/self_condition/result/tiny_imagenet/1G_ACLASS',
        # default='/data/huangcb/self_condition/output/tiny_front_10_GDC/conditional',
        default='/data/huangcb/self_condition/output/tiny_front_10/conditional',
        help="path to images"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='tiny',
        choices=['cifar10','tiny']
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        default='',

    )
    args = parser.parse_args()

    models_list = []
    accus = []
    # torch.cuda.set_device(1)
    #源模型 VGG16
    for i in [0]:
        globals()['source_model' + str(i)] = load_model(i, "source_model",args.dataset)
        models_list.append(globals()['source_model' + str(i)])

    # 模型提取-概率 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['extract_p' + str(i)] = load_model(i,"extract_p",args.dataset)
        models_list.append(globals()['extract_p' + str(i)])

    #模型提取-标签 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['extract_l' + str(i)] = load_model(i,"extract_l",args.dataset)
        models_list.append(globals()['extract_l' + str(i)])
    #
    # #模型提取-对抗 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['extract_adv' + str(i)] = load_model(i, "extract_adv",args.dataset)
        models_list.append(globals()['extract_adv' + str(i)])

    # 剪枝 VGG 15个
    for i in range(5):
        # if i>=10:
        globals()['fine-pruning' + str(i)] = load_model(i, "fine-pruning", args.dataset).cuda()
        models_list.append(globals()['fine-pruning' + str(i)])

    #微调f-a/f-l VGG 20个
    for i in range(20):
        globals()['finetune' + str(i)] = load_model(i, 'finetune',args.dataset)
        models_list.append(globals()['finetune' + str(i)])

    # #迁移学习 VGG 10个
    for i in range(10):
         globals()['transfer' + str(i)] = load_model(i, "transfer",args.dataset)
         models_list.append(globals()['transfer' + str(i)])
    #
    # #迁移学习 irrelevant  VGG16 resnet18 各5个
    for i in range(10):
         globals()['transfer_irrelevant' + str(i)] = load_model(i, "transfer_irrelevant",args.dataset)
         models_list.append(globals()['transfer_irrelevant' + str(i)])

    # 无关模型 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['irrelevant' + str(i)] = load_model(i, "irrelevant",args.dataset)
        models_list.append(globals()['irrelevant' + str(i)])
    ckptdir = [
        # 'd0.1alpha1.0beta0.0gamma1.0omega5.0',#50-320
    #     'd0.1alpha1.0beta0.0gamma1.0omega2.0',#50-330
    #     'd0.1alpha1.0beta0.0gamma1.0omega3.0',#50-330
    #     'd0.1alpha1.0beta0.0gamma1.0omega4.0',#50-330
        # 'd0.6alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.7alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.8alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.9alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.3alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.4alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.5alpha1.0beta0.0gamma1.0omega1.0',#0-330
        # 'd0.6alpha1.0beta0.0gamma1.0omega1.0',#0-330
    #     'd0.1alpha1.0beta0.0gamma1.0omega6.0',#50-330
    #     'd0.1alpha1.0beta0.0gamma1.0omega8.0',#50-330
    #     'd0.1alpha1.0beta3.0gamma1.0omega5.0',  # 50-230
    #     'd0.1alpha1.0beta4.0gamma1.0omega5.0',  # 50-
    #     'd0.1alpha1.0beta5.0gamma1.0omega5.0',  # 50-
    #     'd0.0alpha0.5beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.0alpha0.7beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.0alpha0.9beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.1alpha1.0beta3.0gamma1.0omega5.0',  # 50-
    #     'd0.1alpha1.0beta4.0gamma1.0omega5.0',  # 50-
    #     'd0.1alpha1.0beta5.0gamma1.0omega5.0',  # 50-
    #     'd0.0alpha0.2beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.0alpha0.3beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.0alpha0.6beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.0alpha0.8beta30.0gamma0.5omega0.0',  # 50-
    #     'd0.1alpha1.0beta0.5gamma1.0omega5.0',  # 50-510
    #     'd0.1alpha1.0beta2.0gamma1.0omega5.0',  # 50-
    #     'd0.1alpha1.0beta1.0gamma0.5omega5.0',  # 50-470
    #     'd0.1alpha1.0beta1.0gamma0.7omega5.0',  # 50-470
    #     'd0.1alpha1.0beta1.0gamma2.0omega5.0',  # 50-470
    #     'd0.1alpha1.0beta1.0gamma3.0omega5.0',  # 50-470
    #     'd0.1alpha1.0beta1.0gamma4.0omega5.0',  # 50-470
    #     'd0.1alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.2alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.3alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.4alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.5alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.6alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.7alpha1.0beta0.0gamma1.0omega5.0',  # 50-320
    #     'd0.5alpha1.0beta0.0gamma1.0omega1.0',  # 50-180
    #     'd0.5alpha1.0beta0.0gamma1.0omega0.5',  # 50-320
    #     'd0.5alpha1.0beta0.0gamma1.0omega2.0',  # 50-320
    #     'd0.5alpha1.0beta0.0gamma1.0omega3.0',  # 50-320
    #     'd0.5alpha1.0beta0.0gamma1.0omega5.0',  # 50-500
    #     'd0.2alpha1.0beta0.0gamma1.0omega1.0',#50-400
    #     'd0.3alpha1.0beta0.0gamma1.0omega1.0',
    #     'd0.4alpha1.0beta0.0gamma1.0omega1.0',
    #     'd0.5alpha1.0beta0.0gamma1.0omega1.0',
    #     'd0.6alpha1.0beta0.0gamma1.0omega1.0',
    #     'd0.7alpha1.0beta0.0gamma1.0omega1.0',
    #     'c0.1_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.2_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.3_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.4_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.5_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.6_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.7_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.8_a0.5_g0.5_o10.0_class10',#1-160
    #     'c0.9_a0.5_g0.5_o10.0_class10',#1-160
    #     'd0.1alpha0.2beta0.0gamma1.0omega5.0',#1-160
    #     'd0.1alpha0.4beta0.0gamma1.0omega5.0',#1-160
    #     'd0.1alpha0.6beta0.0gamma1.0omega5.0',#1-160
    #     'd0.1alpha0.7beta0.0gamma1.0omega5.0',#1-160
    #     'd0.1alpha0.8beta0.0gamma1.0omega5.0',#1-160
    #     'd0.1alpha0.9beta0.0gamma1.0omega5.0',#1-160
        # 'd0.1alpha0.2beta0.0gamma1.0omega5.0',#1-160
        # 'd0.1alpha1.0beta1.0gamma1.0omega5.0',
        # 'd0.1alpha1.0beta2.0gamma1.0omega5.0',
        # 'd0.1alpha1.0beta3.0gamma1.0omega5.0',
        # 'd0.1alpha1.0beta4.0gamma1.0omega5.0',
        # 'd0.1alpha1.0beta5.0gamma1.0omega5.0',
        # 'd0.1alpha1.0beta6.0gamma1.0omega5.0'
    ]
    ###写入excel表
    for ckpt in ckptdir:
        args.img_path=os.path.join(args.root_path,ckpt, 'imgs')
        print(ckpt)
        workbook = xw.Workbook(f'{ckpt}.xlsx')  # 创建工作簿
        worksheet1 = workbook.add_worksheet()  # 创建子表
        worksheet1.activate()  # 激活表
        row = 1
        auc_result(args,models_list,worksheet1,row)
        workbook.close()

    # auc_pickle(ip_erase="original")
    # auc_pickle(ip_erase="query_attack")
    # auc_pickle(ip_erase="input_smooth")
    # auc_pickle(ip_erase="squeeze_colorbit")
    #
    # auc_pickle(ip_erase="mixup_0.1")
    # auc_pickle(ip_erase="mixup_0.2")
    # auc_pickle(ip_erase="mixup_0.3")
    # auc_pickle(ip_erase="mixup_0.4")
    # auc_pickle(ip_erase="mixup_0.5")
    # auc_pickle(ip_erase="mixup_0.6")
    # auc_pickle(ip_erase="mixup_0.7")
    # auc_pickle(ip_erase="mixup_0.8")
    # auc_pickle(ip_erase="mixup_0.9")
    # auc_pickle(ip_erase="mixup_1.0")

# pro: 0.94375 lab: 0.915 adv: 0.625 fp: 0.8 ft: 1.0 cif: 0.85
# pro: 0.7825 lab: 0.6212500000000001 adv: 0.27125 fp: 0.7 ft: 1.0 cif: 0.75
# pro: 0.7575 lab: 0.44625 adv: 0.23749999999999996 fp: 0.76 ft: 0.7575000000000001 cif: 0.31499999999999995
# pro: 0.98125 lab: 0.995 adv: 0.53125 fp: 0.7900000000000001 ft: 1.0 cif: 0.9400000000000001
