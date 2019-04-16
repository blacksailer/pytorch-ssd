import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([125, 124, 1])  # RGB layout
image_std = 128.0
iou_threshold = 0.6
center_variance = 0.1
size_variance = 0.1

specs = [
    SSDSpec(19, 16, SSDBoxSizes(10, 30), [2, 1]),
    SSDSpec(10, 32, SSDBoxSizes(30, 68), [2, 1]),
    #  SSDSpec(5, 64, SSDBoxSizes(68, 128), [2, 3]),
    # SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    # SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    # SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)