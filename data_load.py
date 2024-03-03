from dataset import dataset
import utils
from torchvision import transforms,datasets
def load_dataset(name):
    if name== 'cifar10':
        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(utils.CIFAR_MEAN, utils.CIFAR_STD)])
        train_data = dataset("dataset_defend_1.h5", train=False)
        test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test
        )
    elif name== 'tiny':
        transform_train = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.4804, 0.4482, 0.3976], std=[0.2770, 0.2691, 0.2822])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.4804, 0.4482, 0.3976], std=[0.2770, 0.2691, 0.2822])

        ])

        train_data=datasets.ImageFolder('data/extend',transform=transform_train)
        test_data=datasets.ImageFolder('data/val',transform=transform_test)
    return train_data,test_data
