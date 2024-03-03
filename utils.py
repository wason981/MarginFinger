import time
import random
import numpy as np
import os
import torch
import pickle as pkl
import matplotlib.pyplot as plt
from torchvision import  transforms,datasets
import torch.nn.functional as F
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
MEAN_STD = {'tiny':{'MEAN':(0.485, 0.456, 0.406),'STD':(0.229, 0.224, 0.225)},
            'cifar10':{'MEAN':(0.4914, 0.4822, 0.4465),'STD':(0.2023, 0.1994, 0.2010)}}
TINY_MEAN=[0.485, 0.456, 0.406]
TINY_STD=[0.229, 0.224, 0.225]
class CustomDataset(Dataset):
    def __init__(self, data_dict_list):
        self.data = data_dict_list['data']
        self.label = data_dict_list['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx])

def timer(func: callable) -> callable:
    """A timer decorator to time the execution of a function.

    Args:
        func (_type_): Functions that require timing.
    Returns:
        function: The decorated function.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start):.2f} seconds to execute.")
        return result

    return wrapper

def seed_everything(seed: int):
    """Set a random seed for reproducibility of results.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_result(path: str, data: object) -> None:
    """Serialize data from memory to local.

    Args:
        path (str): local path, no exist then new path.
        data (object): data waiting to serialize
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    #
    with open(path, mode="wb") as file:
        pkl.dump(obj=data, file=file)
        print(f"save to {path} successfully!")


def load_result(path: str) -> object:
    """Deserialize data from local to memory.

    Args:
        path (str): local path.

    Raises:
        FileNotFoundError: path spell error or unexist.

    Returns:
        object: object
    """
    if not os.path.exists(path):
        print(f"{path} not found error!")
    with open(path, "rb") as file:
        data = pkl.load(file=file)
    return data

def calculate_auc(list_a, list_b, method, mode, show=False):
    """Calculate the area under the ROC curve (AUC)

    Args:
        list_a (_type_): containing the predicted values ​for the samples.
        list_b (_type_): containing the predicted values ​​for the samples.
        mode (_type_): mode description
        show (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: value of auc.
    """
    l1, l2 = len(list_a), len(list_b)
    y_true, y_score = [], []
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    auc(fpr, tpr)
    if show:
        plot_roc(fpr, tpr, auc(fpr, tpr), method, mode)
    return round(auc(fpr, tpr), 2)

def plot_roc(fpr, tpr, auc, method, mode):
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="ROC (AUC = {:.2f})".format(auc),
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="-", linewidth=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    os.makedirs(f"./roc/{method}/", exist_ok=True)
    plt.savefig("./roc/{}/{}.jpg".format(method, mode), dpi=300)
    plt.close()

def test(datatype,model, dataloader, plot: bool = False, trans: bool = False):
    """Test model classification accuracy

    Args:
        model (_type_): model
        dataloader (_type_): dataloader
        device (_type_): device.'gpu' or 'cpu'

    Returns:
        _type_: accuracy, value range 0-1.
    """
    model.eval()
    model. cuda()
    total = len(dataloader.dataset)
    correct = 0
    probs = []
    # class_array = np.array(CIFAR10_CLASS_NAMES)
    for _, batch_data in enumerate(dataloader):
        b_x, b_y = batch_data[0]. cuda(), batch_data[1]. cuda()
        if trans:
            transform = transforms.Normalize(MEAN_STD[datatype]['MEAN'], MEAN_STD[datatype]['STD'])
            b_x = transform(b_x)
        output = model(b_x)
        output = F.softmax(output, dim=1)
        # print(f"output:{output[-10:]}")
        probs.append(output)
        pred = torch.argmax(output, dim=1)
        correct += (pred == b_y).sum().item()
    probs = torch.cat(probs, dim=0).cpu().detach()
    if plot:
        prob_difference(probs=probs, k=5)
    model = model.cpu()
    return round(correct / total, 2)
def prob_difference(probs, k=2, mode="meta_lw"):
    """Draw a line graph of the probability distribution of the model versus sample output.

    Args:
        probs (_type_): Probability distribution of samples, shape(b_size, n_classes).
        k (int, optional): topk components of the probability, defaults to 2.
    """
    x = len(probs)
    y_s = torch.topk(probs, k=k, dim=1)[0]
    for i in range(k):
        plt.plot(
            range(x),
            y_s[:, i],
            color=generate_random_color(),
            label=f"top_prob_{i}",
        )
    plt.fill_between(
        range(x),
        y_s[:, 0],
        y_s[:, 1],
        where=(y_s[:, 0] > y_s[:, 1]),
        color=generate_random_color(),
        alpha=0.3,
    )
    plt.title(f"top {k} prob")
    plt.xlabel("sample number")
    plt.ylabel("prob")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    save_dir = "./ours/result/db_prob/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{mode}.pdf"), dpi=300)

def generate_random_color():
    """randomly return a color.

    Returns:
        _type_: RGB color
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = (r / 255, g / 255, b / 255)
    return color

def images_to_pickle(input_path='input_path',output_filename='images.pkl'):
    # 存储图像数据的列表 #不归一化
    images_data = []
    for image_file in os.listdir(input_path):
        image_path = os.path.join(input_path, image_file)
        image = Image.open(image_path)
        images_data.append(transforms.ToTensor()(image))
        
    datas=torch.stack(images_data)
    labels =torch.stack([torch.zeros_like(datas)]).squeeze()

    save_result(output_filename, {"data": datas,'label':labels})


def denormalize(batch_data: torch.Tensor):
    transform_reverse = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in TINY_STD]),
            transforms.Normalize(mean=[-m for m in TINY_MEAN], std=[1, 1, 1]),
        ]
    )
    data = transform_reverse(batch_data)
    data = torch.clamp(input=data, min=0.0, max=1.0)
    return data

def normalize(batch_data: torch.Tensor):
    normal = transforms.Normalize(mean=TINY_MEAN, std=TINY_STD)
    data = normal(batch_data)
    return data


def ae_dataset(
    num: int = 500, save_path: str = "./fingerprint/query_attack/ae_dataset.pkl"
):
    seed_everything(2023)
    transform = transforms.ToTensor()
    # dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    dataset = datasets.ImageFolder('data/val__', transform=transform)

    sampled_data = []
    for class_index in range(10):
        class_indices = torch.where(torch.tensor(dataset.targets) == class_index)[0]
        sampled_indices = torch.randperm(len(class_indices))[:num]
        sampled_data.extend([dataset[i] for i in class_indices[sampled_indices]])
    datas = torch.stack([sample[0] for sample in sampled_data])
    labels = torch.stack([torch.tensor(sample[1]) for sample in sampled_data])
    save_result(save_path, {"data": datas, "label": labels})

import torch
from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn.functional import softmax
import numpy as np
# from scipy.stats import entropy
from gan_training.metrics import inception_score
def calculate_inception_score(method, splits=10):
    data_set = load_result(
        f"./fingerprint/original/{method}_fingerprint.pkl"
    )
    if method  in ['cem','ipguard','sac_m','sac_w']:
        data_set['data']=denormalize(data_set['data'])
    score, score_std=inception_score(data_set['data'],
                    device='cuda:0',
                    resize=True,
                    batch_size=20,
                    splits=1)
    # dataset = CustomDataset(data_set)
    # dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False)

    # 转换图像维度和范围
    # images = images.transpose(0, 3, 1, 2)  # 调整维度为 (N, C, H, W)
    # images = images / 255.0  # 将范围缩放到 [0, 1]

    # 转换为 PyTorch 张量
    # images_tensor = torch.tensor(images, dtype=torch.float32)

    # 加载 Inception 模型
    # inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    # inception_model.eval()

    # 数据加载器
    # dataloader = torch.utils.data.DataLoader(images_tensor, batch_size=batch_size)

    # 存储每个批次的概率分布
    # all_preds = []

    # 遍历数据加载器
    # for batch in dataloader:
    #     batch = batch[0].cuda()
    #     with torch.no_grad():
    #         # 提取特征
    #         preds, _ = inception_model(batch)
    #
    #         # 计算 softmax
    #         preds = softmax(preds, dim=1)
    #
    #         # 存储概率分布
    #         all_preds.append(preds.cpu().numpy())
    #
    # # 合并所有批次的概率分布
    # all_preds = np.concatenate(all_preds, axis=0)

    # 计算每个图像的平均 KL 散度
    # scores = []
    # for i in range(splits):
    #     part = all_preds[
    #         (i * all_preds.shape[0] // splits):((i + 1) * all_preds.shape[0] // splits),
    #         :]
    #     kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), axis=0)))
    #     kl = np.mean(np.sum(kl, axis=1))
    #     scores.append(np.exp(kl))

    # 计算最终的 Inception Score
    # is_score = np.mean(scores)
    # print(/ score

# images_to_pickle('output/tiny_front_10/conditional/d0.1alpha1.0beta0.0gamma1.0omega5.0/imgs/320/samples','fingerprint/original/margin_fingerprint.pkl')
# calculate_inception_score('cem')
# calculate_inception_score('ipguard')
# calculate_inception_score('margin')
# calculate_inception_score('sac_m')
# calculate_inception_score('sac_w')
# calculate_inception_score('margin')
# calculate_inception_score('mixup_0.1')
# calculate_inception_score('mixup_0.2')
# calculate_inception_score('mixup_0.3')
# calculate_inception_score('mixup_0.4')
# calculate_inception_score('mixup_0.5')
# calculate_inception_score('mixup_0.6')
# calculate_inception_score('mixup_0.7')
# calculate_inception_score('mixup_0.8')
# calculate_inception_score('mixup_0.9')
