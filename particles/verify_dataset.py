import os
import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from datasets.SyntheticParticlesDataset import SyntheticParticlesDataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt
dataset_path = 'D:/datasets/Bezel_Full/PASCAL_VOC'

dataset_path = '../Data/Bezel_Gray/PASCAL_VOC'
dataset = SyntheticParticlesDataset(dataset_path)
# ,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((300,300)),torchvision.transforms.ToTensor()])),transform=train_transform, target_transform=target_transform
loader = DataLoader(dataset, 1,
                    num_workers=0,
                    shuffle=False)
img_mean = np.zeros((300,300,3))
# for i, data in enumerate(loader):
#     images, boxes, labels, ids = data
#     orig_image = cv2.imread(dataset_path + '/JPEGImages/{}.jpg'.format(ids[0]))
#     orig_image = cv2.transpose(orig_image)
#     cv2.imwrite(dataset_path + '/JPEGImages/{}.jpg'.format(ids[0]), orig_image)

for i, data in enumerate(loader):
    images, boxes, labels, ids = data
    orig_image = cv2.imread(dataset_path + '/JPEGImages/{}.jpg'.format(ids[0]))
    #orig_image = orig_image.T
    print(ids[0])
    img_mean += cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    print(orig_image.shape)

    #orig_image = cv2.transpose(orig_image)
    print(orig_image.shape)
    path = "verify.jpg"

    cv2.imwrite(path, orig_image)

    # width, height, _ = image.shape
    for box in boxes[0]:
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3] )), (255, 255, 0), 4)
    #     #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    #     # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    #     # cv2.putText(orig_image, label,
    #     #             (box[0] + 20, box[1] + 40),
    #     #             cv2.FONT_HERSHEY_SIMPLEX,
    #     #             1,  # font scale
    #     #             (255, 0, 255),
    #     #             2)  # line type
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',orig_image.T)
    cv2.waitKey()
    path = "verify.jpg"
   # cv2.imwrite(path, orig_image)
    break
# print(np.mean(img_mean/179, axis=(0,1)))
# print(np.std(img_mean/179, axis=(0,1,2)))