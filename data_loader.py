import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import Dataset
import h5py
import os
from dataset import dataset
import numpy as np
from PIL import Image

transform_tiny = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((64,64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class Data_set(Dataset):
    def __init__(self, name,train=False):
        super(Data_set, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("/home/liuwx/Data/sac/data",name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        # image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = np.array(self.images[item, :, :, :])
        image = transform_train(image)
        return [image,label]


class Data_Loader():
    def __init__(self, batch_size,type, shuf=True):
        self.batch = batch_size
        self.shuf = shuf
        self.type=type
    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset
    def load_cifar10(self,class_):
        train_data = Data_set('dataset_defend_1.h5', train=False)
        if self.type=='1G1C':
            train_data.images = train_data.images[train_data.labels == class_]
            train_data.labels = train_data.labels[train_data.labels == class_]
        return train_data

    def load_tiny(self):
        train_data = dsets.ImageFolder(root='/data/huangcb/self_condition/data/front-10',
                                                      transform=transform_tiny)

        return train_data
    def loader(self,class_,dataset):
        if dataset=='cifar10':
            train_data = self.load_cifar10(class_)
            loader = torch.utils.data.DataLoader(dataset=train_data,
                                                 batch_size=self.batch,
                                                 shuffle=self.shuf,
                                                 num_workers=0,
                                                 drop_last=True)
        else:
            train_data = self.load_tiny()
            if self.type == '1G1C':
                from torch.utils.data import Subset
                indice=[i for i in range(len(train_data)) if train_data.targets[i] == class_]
                selected_dataset=Subset(train_data,indice)
                # class_to_load = os.listdir('/home/liuwx/Data/data/front-100/defender/train/')[class_]
                # class_index = train_data.class_to_idx[class_to_load]
                # class_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                #     [i for i in range(len(train_data)) if train_data.targets[i] == class_])
                loader=torch.utils.data.DataLoader(dataset=selected_dataset,
                                                   batch_size=self.batch,
                                                   num_workers=0,
                                                   drop_last=True,
                                                   )
            else :
                loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=self.batch,
                                                  shuffle=self.shuf,
                                                  num_workers=0,
                                                  drop_last=True)
        return loader

