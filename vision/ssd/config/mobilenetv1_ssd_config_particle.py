import numpy as np

#from vision.utils.circle_utils import SSDSpec, SSDBoxSizes,  generate_ssd_circle_priors
from vision.utils.box_utils import SSDSpec, SSDBoxSizes,  generate_ssd_priors


image_size = 300
#image_mean = np.array([74, 62, 44])  # RGB layout
image_mean = np.array([ 62])  # RGB layout

image_std = 13.0
iou_threshold = 0.6
center_variance = 0.1
size_variance = 0.1

specs = [
    # SSDSpec(38, 16, SSDBoxSizes(10, 30), [2, 1]),
    SSDSpec(38, 2**3, SSDBoxSizes(14, 16),  [1, 1]),
    #  SSDSpec(19, 64, SSDBoxSizes(10, 68), [2, 1]),
    SSDSpec(19, 2**4, SSDBoxSizes(18, 22), [1,  2]),
    # SSDSpec(10, 150, SSDBoxSizes(10, 68), [2, 1]),
    # SSDSpec(10, 300, SSDBoxSizes(10, 68), [2, 1])
]


priors = generate_ssd_priors(specs, image_size)
#priors = generate_ssd_circle_priors(specs, image_size)