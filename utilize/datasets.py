import os
import random
from tqdm import *
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def tensorshow(tensors, titles=None):
    '''
    BCWH
    '''
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211+i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


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
    def __init__(self, path, train, transform, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.format = format
        self.transform = transform
        self.train = train

        haze_img_dir = os.path.join(path, 'hazy')
        self.clea_img_dir = os.path.join(path, 'gt')

        self.haze_file_path = sorted(get_file_paths(haze_img_dir))

        self.n_samples = len(self.haze_file_path)

    def __getitem__(self, idx):
        # img_input, img_target = self.get_img_pair(idx)
        img_haze = Image.open(self.haze_file_path[idx]).convert('RGB')
        # print("haze:", self.haze_file_path[idx])
        img_clea_num = self.haze_file_path[idx].split('_')[0].split('\\')[-1]
        # print("img_clea_num:", img_clea_num)
        img_clea = Image.open(os.path.join(self.clea_img_dir, img_clea_num) + self.format).convert('RGB')
        # print("gt:", os.path.join(self.clea_img_dir, img_clea_num) + self.format)
        if self.train:
            img_haze, img_clea = augment(img_haze, img_clea)
        img_haze, img_clea = self.transform(img_haze), self.transform(img_clea)

        return img_haze, img_clea

    def __len__(self):
        return self.n_samples

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])


path = 'D:/zwx/zwx/dehaze/dataset/'
batch_size = 4
num_workers = 8

# RESIDE数据集
ITS_train_loader = torch.utils.data.DataLoader(
    dataset=RESIDE_Dataset(path+'RESIDE/ITS/train', train=True, transform=transform, format='.png'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
ITS_val_loader = torch.utils.data.DataLoader(
    dataset=RESIDE_Dataset(path+'RESIDE/ITS/val', train=False, transform=transform, format='.png'),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)
OTS_train_loader = torch.utils.data.DataLoader(
    dataset=RESIDE_Dataset(path+'RESIDE/OTS', train=True, transform=transform, format='.jpg'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
OTS_val_loader = torch.utils.data.DataLoader(
    dataset=RESIDE_Dataset(path+'RESIDE/SOTS/outdoor', train=False, transform=transform, format='.png'),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)


# I-HAZE/O-HAZE CVPR2018
IHAZY_train_loader = torch.utils.data.DataLoader(
    dataset=RESIDE_Dataset(path+'NTIRE/IHAZE/train', train=True, transform=transform, format='.jpg'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
IHAZY_val_loader = torch.utils.data.DataLoader(
    dataset=RESIDE_Dataset(path+'NTIRE/IHAZE/val', train=False, transform=transform, format='.jpg'),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)
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

# if __name__ == "__main__":
#     loop = tqdm(enumerate(ITS_val_loader), total=len(ITS_train_loader))
#     for i, (haze, gt) in loop:
#         # loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
#         loop.set_postfix(loss=i, shape=haze.shape)
#         # tensorshow(haze, titles='haze')
#         # tensorshow(gt, titles='gt')
#         # if i == 2: break


