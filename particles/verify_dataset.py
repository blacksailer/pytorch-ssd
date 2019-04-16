import os
import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from datasets.SyntheticParticlesDataset import SyntheticParticlesDataset
import torchvision

dataset_path = '../Data/train/PASCAL_VOC'


dataset = SyntheticParticlesDataset(dataset_path)
# ,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((300,300)),torchvision.transforms.ToTensor()])),transform=train_transform, target_transform=target_transform
loader = DataLoader(dataset, 1,
                    num_workers=0,
                    shuffle=False)
for i, data in enumerate(loader):
    images, boxes, labels, ids = data
    orig_image = cv2.imread(dataset_path + '/JPEGImages/{}.jpg'.format(ids[0]))
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    width, height, _ = image.shape
    for box in boxes[0]:
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3] )), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # cv2.putText(orig_image, label,
        #             (box[0] + 20, box[1] + 40),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,  # font scale
        #             (255, 0, 255),
        #             2)  # line type
    path = "verify.jpg"
    cv2.imwrite(path, orig_image)
    break