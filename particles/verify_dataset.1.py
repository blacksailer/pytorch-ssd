import os
import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from datasets.SyntheticParticlesDataset import SyntheticParticlesDataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt
dataset_path = 'D:/datasets/Bezel/PASCAL_VOC'

dataset_path = '../Data/Mixin/Video_1299/PASCAL_VOC'
dataset = SyntheticParticlesDataset(dataset_path)
# ,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((300,300)),torchvision.transforms.ToTensor()])),transform=train_transform, target_transform=target_transform
loader = DataLoader(dataset, 1,
                    num_workers=0,
                    shuffle=False)
img_mean = np.zeros((300,300,3))
results = []
for i, data in enumerate(loader):
    images, boxes, labels, ids = data
    if boxes.shape[1] != 0:
        results.append(ids[0])

with open(os.path.join(dataset_path,'ImageSets','Main','train.txt'),'w') as f:
    f.write("\n".join(results))
print(len(results),results)
# print(np.mean(img_mean/179, axis=(0,1)))
# print(np.std(img_mean/179, axis=(0,1,2)))