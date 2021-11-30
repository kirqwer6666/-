from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision
import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

def get_file_paths(folder):
    file_paths = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        file_paths.append(file_path)
    file_paths = sorted(file_paths)
    return file_paths

def augment(img_input, img_target):
    degree = random.choice([0, 90, 180, 270])
    if degree != 0:
        img_input = transforms.functional.rotate(img_input, degree)
        img_target = transforms.functional.rotate(img_target, degree)

    return img_input, img_target

class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, transform, format='.jpg'):
        super(RESIDE_Dataset, self).__init__()
        self.format = format
        self.transform = transform
        self.train = train

        haze_img_dir = os.path.join(path, 'hazy')
        self.clea_img_dir = os.path.join(path, 'gt')

        self.haze_file_path = get_file_paths(haze_img_dir)

        self.n_samples = len(self.haze_file_path)

    def get_img_pair(self, idx):
        img_haze = Image.open(self.haze_file_path[idx]).convert('RGB')
        img_clea = self.haze_file_path[idx].split('_')[1].split('/')[-1] + self.format
        img_clea = Image.open(os.path.join(self.clea_img_dir, img_clea)).convert('RGB')

        return img_haze, img_clea

    def __getitem__(self, idx):
        img_input, img_target = self.get_img_pair(idx)
        if self.train:
            img_input, img_target = augment(img_input, img_target)
        img_input, img_target = self.transform(img_input), self.transform(img_target)

        return img_input, img_target

    def __len__(self):
        return self.n_samples

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


path = './datasets/'
batch_size = 1
num_workers = 8

# ITS_train_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'RESIDE/ITS', train=True, transform=transform),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers)
# ITS_val_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'RESIDE/SOTS/indoor', train=False, transform=transform),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers)
# OTS_train_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'RESIDE/OTS', train=True, transform=transform),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers)
# OTS_val_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'RESIDE/SOTS/outdoor', train=False, transform=transform),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers)
#
# IHAZY_train_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'NTIRE/IHAZY', train=True, transform=transform),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers)
# IHAZY_val_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'NTIRE/TEST/indoor', train=False, transform=transform),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers)
# OHAZY_train_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'NTIRE/OHAZY', train=True, transform=transform),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers)
# OHAZY_val_loader = torch.utils.data.DataLoader(
#     dataset=RESIDE_Dataset(path+'NTIRE/TEST/outdoor', train=False, transform=transform),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers)

# DENSE-HAZE CVPR2019
# NH-HAZE CVPR2020

class Test_Dataset(data.Dataset):
    def __init__(self, path, transform):
        super(Test_Dataset, self).__init__()
        self.format = format
        self.transform = transform

        self.file_path = get_file_paths(path)

        self.n_samples = len(self.file_path)

    def __getitem__(self, idx):
        img = Image.open(self.file_path[idx]).convert('RGB')
        img_input= self.transform(img)
        return img_input

    def __len__(self):
        return self.n_samples

TestLoader = torch.utils.data.DataLoader(
    dataset=Test_Dataset(path='./ok', transform=transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

def tensorshow(tensors):
    '''
    BCWH
    '''
    fig = plt.figure()
    for tensor, i in zip(tensors, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.detach().numpy()
        ax = fig.add_subplot(211+i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




