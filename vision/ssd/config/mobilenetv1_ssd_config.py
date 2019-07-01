import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(10, 20), [2, 1]),
    SSDSpec(10, 32, SSDBoxSizes(20, 30), [2, 1]),
    SSDSpec(5, 64, SSDBoxSizes(30, 40), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(40, 50), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(50, 60), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(60, 70), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)