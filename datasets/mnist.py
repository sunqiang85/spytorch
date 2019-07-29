import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision import transforms


class dataset(data.Dataset):

    def __init__(self, config,  split='train'):
        self.split = split
        self.dir_data = config.dir_data

        if self.split == 'train':
            is_train = True
            self.shuffle = True
        elif self.split == 'val':
            is_train = False
            self.shuffle = False
        else:
            raise ValueError()

        self.item_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        download = (not os.path.isdir(self.dir_data))
        self.dataset = datasets.MNIST(self.dir_data,
            train=is_train, download=download, transform=self.item_tf)

        self.dataset = datasets.MNIST(self.dir_data,
            train=is_train, download=download, transform=self.item_tf)


    def __getitem__(self, index):
        data, class_id = self.dataset[index]
        item = {}
        item['index'] = index
        item['data'] = data
        item['class_id'] = torch.LongTensor([class_id])
        return item

    def __len__(self):
        return len(self.dataset)


class CompositeDataset(Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))

def get_data_loader(config, split):
    if split == 'trainval':
        dset1 = dataset(config, 'train')
        dset2 = dataset(config, 'val')
        dset = CompositeDataset(dset1, dset2)
    else:
        dset = dataset(config, split)

    if split == 'train' or split == 'trainval':
        shuffle = True
    else:
        shuffle = False

    data_loader = data.DataLoader(dset,
        batch_size=config.batch_size,
        num_workers=config.nb_threads,
        shuffle=shuffle,
        pin_memory=config.pin_memory,
        #pin_memory=True,
        drop_last=False)
    return data_loader