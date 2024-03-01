import h5py
import os
import numpy as np
import torch
from torchvision import transforms

transform_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

class dataset(torch.utils.data.Dataset):
    def __init__(self, name, train=False):
        super(dataset, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("../data", name), "r")
        self.images = np.array(self.data["/data"])
        self.labels = np.array(self.data["/label"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype="uint8")
        image = transform_train(image)
        return [image, label]
