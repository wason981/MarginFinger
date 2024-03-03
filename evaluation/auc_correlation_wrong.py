import torch
from dataset import dataset1,dataset4
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc,roc_auc_score
from model_load import load_model
import matplotlib.pyplot as plt
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
BATCH_SIZE = 100

class FeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


def calculate_auc(list_a, list_b, mode):
    l1, l2 = len(list_a), len(list_b)
    y_true, y_score = [], []
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)###真实标签[0,0,0,0,0,1,1,1,1,1]
    y_score.extend(list_a)
    y_score.extend(list_b)###预测分数(概率）[0.09,0.08,0.08,... 0.99,0.98,0.97]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)###设置不同的阈值，根据预测分数与不同阈值画出roc
    auc(fpr, tpr)
    plot_roc(fpr, tpr, auc(fpr, tpr), mode)
    return auc(fpr, tpr)


def plot_roc(fpr, tpr, auc, mode):
    plt.plot(fpr, tpr, "k--", label="ROC (area={0:.2f})".format(auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("./roc/SAC/{}.jpg".format(mode))
    plt.close


# consine similarity
def correlation(m, n):
    m = F.normalize(m, dim=-1)
    n = F.normalize(n, dim=-1).transpose(0, 1)
    cose = torch.mm(m, n)
    matrix = 1 - cose
    matrix = matrix / 2
    return matrix


def pairwise_euclid_distance(A):
    sqr_norm_A = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=0)
    sqr_norm_B = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=1)
    inner_prod = torch.matmul(A, A.transpose(0,1))
    tile1 = torch.reshape(sqr_norm_A,[A.shape[0],1])
    tile2 = torch.reshape(sqr_norm_B,[1,A.shape[0]])
    return tile1+tile2 - 2*inner_prod


def correlation_dist(A):
    A = F.normalize(A, dim=-1)
    cor = pairwise_euclid_distance(A)
    cor = torch.exp(-cor)

    return cor


def cal_cor(model,dataloader):##白盒验证
    model.eval()
    model = model.cuda()
    outputs = []
    for i,(x,y) in enumerate(dataloader):
        # if i!=0:
        #    break

        x, y = x.cuda(), y.cuda()
        output = model(x)
        outputs.append(output.cpu().detach())

    output = torch.cat(outputs,dim=0)
    cor_mat = correlation(output,output)

    # cor_mat = correlation_dist(output)

    model = model.cpu()
    return cor_mat

def cal_cor_onehot(model,dataloader):###黑盒验证
    model.eval()
    model = model.cuda()
    outputs = []
    for i,(x,y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        output = output_to_label(output)
        outputs.append(output.cpu().detach())
    output = torch.cat(outputs,dim=0)
    cor_mat = correlation(output,output)
    model = model.cpu()
    return cor_mat

def output_to_label(output):
    shape = output.shape
    pred = torch.argmax(output,dim=1)
    preds = 0.01 * torch.ones(shape)

    for i in range(shape[0]):
        preds[i,pred[i]]=1

    preds = torch.softmax(preds,dim=-1)

    # print(preds[0,:])
    return preds




def cal_correlation(models,i):

    # SAC-normal
    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # train_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # SAC-w
    # train_data = dataset1("dataset_common.h5", train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-w origin
    data_set = utils.load_result("./fingerprint/original/sac_w_fingerprint.pkl")#data_set["data"]=torch.stack(data_set["data"])
    train_data = TensorDataset(data_set["data"], data_set["label"])
    wotrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-w queryattack
    # data_set = utils.load_result("./fingerprint/query_attack/sac_w_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # wqtrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-w input_smoothing
    # data_set = utils.load_result("./fingerprint/input_smooth/sac_w_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # witrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # # SAC-w squeeze_colorbit
    # data_set = utils.load_result("./fingerprint/squeeze_colorbit/sac_w_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # wstrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-m (with flip)
    # train_data = dataset1('cut_mix_100.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-m (without flip)
    # train_data = dataset1('cut_mix_100_nf.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-m (50 samples)
    # train_data = dataset1('cut_mix_50.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-source
    # train_data = dataset1('source_wrong_final.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # sac-m origin
    # data_set = utils.load_result("./fingerprintsac/original/sac_m_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # motrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # sac-m queryattack
    # data_set = utils.load_result("./fingerprint/query_attack/sac_m_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # mqtrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # sac-m input_smoothing
    # data_set = utils.load_result("./fingerprint/input_smooth/sac_m_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # mitrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # sac-m squeeze_colorbit
    # data_set = utils.load_result("./fingerprint/squeeze_colorbit/sac_m_fingerprint.pkl")
    # train_data = TensorDataset(data_set["data"], data_set["label"])
    # mstrain_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    for train_loader in [wotrain_loader]:
        cor_mats = []

        for i in range(len(models)):
            model = models[i]
            cor_mat = cal_cor(model, train_loader)
            cor_mats.append(cor_mat)

        print(len(cor_mats), cor_mat.shape)

        diff = torch.zeros(len(models)-1)

        ###计算与教师模型的 distance
        for i in range(len(models) - 1):
            iter = i + 1
            diff[i] = torch.sum(torch.abs(cor_mats[iter] - cor_mats[0])) / (cor_mat.shape[0] * cor_mat.shape[1])
            # print(cor_mat.shape[0],cor_mat.shape[1])

        # print("Correlation difference is:", diff[:20])
        # print("Correlation difference is:", diff[20:40])
        # print("Correlation difference is:", diff[40:60])
        # print("Correlation difference is:", diff[60:70])
        # print("Correlation difference is:", diff[70:80])
        # print("Correlation difference is:", diff[80:90])
        # print("Correlation difference is:", diff[90:100])
        # print("Correlation difference is:", diff[100:120])
        # print("Correlation difference is:", diff[120:135])
        # print("Correlation difference is:", diff[135:155])

        list1 = diff[:20]
        list2 = diff[20:40]
        list3 = diff[40:60]
        list4 = diff[60:65]
        list5 = diff[65:85]
        list6 = diff[85:95]
        list7 = diff[95:105]
        list8 = diff[105:125]
        # list9 = diff[120:135]
        # list10 = diff[135:155]

        auc_p = calculate_auc(list1, list8, "p")####extract-p 和 无关
        auc_l = calculate_auc(list2, list8, "l")####extract-l
        auc_adv = calculate_auc(list3, list8, "adv")####extract-adv
        auc_prune = calculate_auc(list4, list8, "prune")###pruning
        auc_finetune = calculate_auc(list5, list8, "finetune")###finetune
        auc_tl = calculate_auc(list6, list7, "transfer")####transfer-10C

        print("Calculating AUC:")
        print(
            "AUC_P:",
            auc_p,
            "AUC_L:",
            auc_l,
            "AUC_Adv:",
            auc_adv,
            "AUC_Prune:",
            auc_prune,
            "AUC_Finetune:",
            auc_finetune,
            "AUC_10C:",
            auc_tl,
        )


if __name__ == '__main__':




    models = []

    # 源模型 VGG16
    for i in [0]:
        globals()['source_model' + str(i)] = load_model(i, "source_model", 'tiny')
        models.append(globals()['source_model' + str(i)])

    # 模型提取-概率 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['extract_p' + str(i)] = load_model(i, "extract_p", 'tiny')
        models.append(globals()['extract_p' + str(i)])

    # 模型提取-标签 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['extract_l' + str(i)] = load_model(i, "extract_l", 'tiny')
        models.append(globals()['extract_l' + str(i)])

    #模型提取-对抗 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['extract_adv' + str(i)] = load_model(i, "extract_adv", 'tiny')
        models.append(globals()['extract_adv' + str(i)])

    # 剪枝 VGG 5个
    for i in range(5):
        globals()['fine-pruning' + str(i)] = load_model(i, "fine-pruning", 'tiny')#.cuda(1)
        models.append(globals()['fine-pruning' + str(i)])

    # 微调f-a/f-l VGG 20个
    for i in range(20):
        globals()['finetune' + str(i)] = load_model(i, 'finetune', 'tiny')
        models.append(globals()['finetune' + str(i)])

    # #迁移学习 VGG 10个
    for i in range(10):
        globals()['transfer' + str(i)] = load_model(i, "transfer", 'tiny')
        models.append(globals()['transfer' + str(i)])

    # #迁移学习 irrelevant  VGG16 resnet18 各5个
    for i in range(10):
        globals()['transfer_irrelevant' + str(i)] = load_model(i, "transfer_irrelevant", 'tiny')
        models.append(globals()['transfer_irrelevant' + str(i)])

    # 无关模型 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    for i in range(20):
        globals()['irrelevant' + str(i)] = load_model(i, "irrelevant", 'tiny')
        models.append(globals()['irrelevant' + str(i)])

    for i in range(1):
        iter = i
        print("Iter:", iter)
        cal_correlation(models,iter)

#sac_w:original
#AUC_P: 0.24 AUC_L: 0.26 AUC_Adv: 0.00 AUC_Prune: 0.60 AUC_Finetune: 1.00 AUC_10C: 0.68
#sac_w:query_atack
#AUC_P: 0.67 AUC_L: 0.86 AUC_Adv: 0.57 AUC_Prune: 0.80 AUC_Finetune: 1.00 AUC_10C: 0.00
#sac_w:input_smooth
#AUC_P: 0.03 AUC_L: 0.05 AUC_Adv: 0.00 AUC_Prune: 0.60 AUC_Finetune: 1.00 AUC_10C: 0.70
#sac_w:squeeze_colorbit
#AUC_P: 0.15 AUC_L: 0.16 AUC_Adv: 0.00 AUC_Prune: 0.60 AUC_Finetune: 1.00 AUC_10C: 0.60

#sac_m:original
#AUC_P: 0.37 AUC_L: 0.36 AUC_Adv: 0.06 AUC_Prune: 0.60 AUC_Finetune: 1.00 AUC_10C: 0.75
#sac_m:query_atack
#AUC_P: 0.59 AUC_L: 0.77 AUC_Adv: 0.39 AUC_Prune: 0.79 AUC_Finetune: 1.00 AUC_10C: 0.05
#sac_m:input_smooth
#AUC_P: 0.28 AUC_L: 0.28 AUC_Adv: 0.03 AUC_Prune: 0.60 AUC_Finetune: 1.00 AUC_10C: 0.83
#sac_m:squeeze_colorbit
#AUC_P: 0.35 AUC_L: 0.37 AUC_Adv: 0.02 AUC_Prune: 0.60 AUC_Finetune: 1.00 AUC_10C: 0.69





