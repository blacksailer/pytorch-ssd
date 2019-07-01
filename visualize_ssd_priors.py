import cv2
from vision.ssd.config.mobilenetv1_ssd_config import *
import numpy as np
import matplotlib.pyplot as plt

import torch
from vision.utils.circle_utils import iou_of

width = 300
height = 300

black_image = np.zeros((width,height,3),dtype=np.uint8)

def draw_boxes(black_image,priors):
        for idx,b in enumerate(priors):
                box = b.numpy()
                x_left = box[0] - box[2] / 2
                x_right = box[0] + box[2] / 2
                y_left = box[1] - box[3] / 2
                y_right = box[1] + box[3] / 2
                cv2.rectangle(black_image, (int(x_left * width), int(y_left * height)), (int(x_right * width), int(y_right * height)), (255, 255, 0), 1)
                if idx > 1:
                        break


def draw_circles(black_image,priors):
        for idx,b in enumerate(priors):
                box = b.numpy()
                x = box[0] 
                y = box[1]
                r = box[2] 
                cv2.circle(black_image, (int(x * width), int(y * height)), int( r * width) , (255, 255, 0), 1)
                if idx > 12:
                        break

circ1 = np.array([
        [0,0,2],
        [0,0,2],
        [0,0,2]
],
dtype=np.float)
circ2 = np.array([
        [3,0,2],
        [1,0,2],
        [0,0,1]
],dtype=np.float)

a = torch.Tensor(circ1)
b = torch.Tensor(circ2)
print(iou_of(a,b))
#draw_circles(black_image,priors)
#plt.imshow(black_image)
#plt.show()


