import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image


# RGB 图像的标准化
def normalize(image):
    # 数据集RGB各通道的均值和方差
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape)
    # RGB逐通道进行标准化
    for i in range(3):
        img_copy[i, :, :] = (image[i, :, :] - image_data[0, i]) / image_data[1, i]
    return img_copy


transform_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_CIFAR100 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.53561753, 0.48983628, 0.42546818), (0.26656017, 0.26091456, 0.27394977)
        ),
    ]
)

transform_CIFAR10C_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4645897160947712, 0.6514782475490196, 0.5637088950163399),
            (0.18422159112571024, 0.3151505122530825, 0.26127269383599344),
        ),
    ]
)
transform_CIFAR10C_test = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4645897160947712, 0.6514782475490196, 0.5637088950163399),
            (0.18422159112571024, 0.3151505122530825, 0.26127269383599344),
        ),
    ]
)


class dataset(Dataset):
    def __init__(self, name, train=False):
        super(dataset, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype="uint8")
        image = transform_train(image)
        return [image, label]


class dataset1(Dataset):
    def __init__(self, name, train=False):
        super(dataset1, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype="uint8")
        image = transform_test(image)
        return [image, label]


class query_attack_dataset(Dataset):
    def __init__(self, name, device, ae) -> None:
        super().__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.device = device
        self.ae = ae
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype="uint8")
        image = transform_test(image)
        self.ae.eval()
        image = torch.squeeze(
            self.ae(torch.unsqueeze(image, dim=0).to(self.device)), dim=0
        ).cpu()
        return [image, label]


class dataset3(Dataset):
    def __init__(self, name, train=False):
        super(dataset3, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype="uint8")
        image = transform_CIFAR100(image)
        return [image, label]


class dataset4(Dataset):
    def __init__(self, name, train=False):
        super(dataset4, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])
        if train == True:
            self.transform = transform_CIFAR10C_train
        elif train == False:
            self.transform = transform_CIFAR10C_test

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype="uint8")
        image = self.transform(image)
        return [image, label]


class dataset5(Dataset):
    def __init__(self, name, train=False):
        super(dataset5, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        image = np.squeeze(self.images[item, :, :, :])
        image = np.transpose(image, [2, 0, 1])
        image = torch.tensor(image)
        image = normalize(image)
        label = torch.tensor(self.labels[item])
        return [image, label]


class dataset6(Dataset):
    def __init__(self, name, train=False):
        super(dataset6, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        image = np.squeeze(self.images[item, :, :, :])
        image = torch.tensor(image)
        image = normalize(image)
        label = torch.tensor(self.labels[item])
        return [image, label]


class dataset7(Dataset):
    def __init__(self, name, train=False):
        super(dataset7, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        image = np.squeeze(self.images[item, :, :, :])
        image = torch.tensor(image)
        label = torch.tensor(self.labels[item])
        return [image, label]


if __name__ == "__main__":
    # train_data = dataset1("dataset_common.h5", train=False)
    # train_data = dataset1("dataset_common.h5", train=False)
    # datas, labels = [], []
    # for data, label in train_data:
    #     datas.append(data)
    #     labels.append(label)
    # datas = torch.stack(datas, dim=0)
    # labels = torch.tensor(labels)
    # import utils

    # utils.save_result(
    #     "./fingerprint/sac_w_fingerprint.pkl", {"data": datas, "label": labels}
    # )
    # print(train_data[0][0])
    # print(train_data[0][1])
    # print(train_data.data["/data"])
    # print(train_data.images[0])

    data = h5py.File(os.path.join("data", "dataset_common.h5"), "r")
    print(data["/data"].shape)
    print(data["/label"].shape)
