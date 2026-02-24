# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
#from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from conf import base_path,ContinualDataset
from PIL import Image
import os
import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import cv2
from torchvision import transforms
#from datasets.utils.validation import get_train_val
#from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
#from datasets.utils.continual_dataset import get_previous_train_loader
#from datasets.transforms.denormalization import DeNormalize


class Imagenet32(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        root='/home/c206a/zh/CL/OCM-main/OCM-main/data/IMG32'

        if self.train:
            self.data = []
            for num in range(10):
                b=np.load(os.path.join(
                    root, 'Imagenet32_train_npz/x_train_%02d.npy' %
                        (num+1)))
                if num == 9:
                    b = b[:128116, :]
                self.data.append(b)
                
                    
            self.data = np.concatenate(np.array(self.data))

            self.targets = []
            for num in range(10):
                b = np.load(os.path.join(
                    root, 'Imagenet32_train_npz/y_train_%02d.npy' %
                        (num+1)))
                if num == 9:
                    b = b[:128116]
                self.targets.append(b)

            self.targets = np.concatenate(np.array(self.targets))
        else:
            self.data = []

            self.data.append(np.load(os.path.join(
                root, 'Imagenet32_val_npz/x_val.npy')))
            self.data = np.concatenate(np.array(self.data))

            self.targets = []

            self.targets.append(np.load(os.path.join(
                root, 'Imagenet32_val_npz/y_val.npy' )))

            self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyImagenet32(Imagenet32):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None) -> None:
        super(MyImagenet32, self).__init__(
            root, train, transform, target_transform)

    def __getitem__(self, index, img_size=32):
        x, target = self.data[index], self.targets[index]-1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_size2 = img_size * img_size
        x = np.dstack((x[:img_size2], x[img_size2:2*img_size2], x[2*img_size2:]))
        x = x.reshape((img_size, img_size, 3))#.transpose(2, 0, 1)
        img = Image.fromarray((x))
        # img.save('1.jpg')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SequentialImagenet32(ContinualDataset):

    NAME = 'seq-img32'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 1000
    N_TASKS = 1
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])

    def get_data_loaders():
        #transform = self.TRANSFORM
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train_dataset = MyImagenet32(base_path() + 'IMG32',
                                 train=True,  transform=test_transform)
        #if self.args.validation:
         #   train_dataset, test_dataset = get_train_val(train_dataset,
         #                                           test_transform, self.NAME)
        #else:
        test_dataset = MyImagenet32(base_path() + 'IMG32',
                        train=False, transform=test_transform)

        #train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return DataLoader(train_dataset, batch_size=64, shuffle=True),DataLoader(test_dataset, batch_size=64, shuffle=True)#train_dataset, test_dataset

    #def not_aug_dataloader(self, batch_size):
     #   transform = transforms.Compose([transforms.ToTensor(), self.get_denormalization_transform()])

      #  train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                            #train=True, download=True, transform=transform)
       # train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        #return train_loader


    @staticmethod
    #def get_backbone():
     #   return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
      #                  * SequentialTinyImagenet.N_TASKS)

    @staticmethod

    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    #@staticmethod
    #def get_denormalization_transform():
     #   transform = DeNormalize((0.4802, 0.4480, 0.3975),
     #                                    (0.2770, 0.2691, 0.2821))
      #  return transform