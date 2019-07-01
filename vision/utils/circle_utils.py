import collections
import torch
import itertools
from typing import List
import math

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
SSDCircleSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def generate_ssd_circle_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDCircleSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDCircleSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDCircleSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDCircleSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDCircleSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDCircleSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 3): The prior boxes represented as [[center_x, center_y, r]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            r = size / image_size
            priors.append([
                x_center,
                y_center,
                r
            ])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            r = size / image_size
            priors.append([
                x_center,
                y_center,
                r
            ])

            # # change h/w ratio of the small sized box
            # size = spec.box_sizes.min
            # r = size / image_size
            # for ratio in spec.aspect_ratios:
            #     ratio = math.sqrt(ratio)
            #     priors.append([
            #         x_center,
            #         y_center,
            #         r * ratio,
            #     ])
            #     priors.append([
            #         x_center,
            #         y_center,
            #         r / ratio,
            #     ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

def convert_locations_to_circles(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, r).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 3): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 3) or (batch_size/1, num_priors, 3): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        circles:  priors: [[center_x, center_y, r]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_circles_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(radius) -> torch.Tensor:
    """Compute the areas of circles given radius.

    Args:
        radius (N, 1): left top corner

    Returns:
        area (N): return the area.
    """
    hw = math.pi * radius * radius
    return hw


def iou_of(circles0, circles1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        circles0 (M, 3): ground truth boxes.
        circles1 (N or 1, 3): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (M): IoU values.
    """
    result = torch.zeros(circles0.shape[0],)

    
    d = torch.sqrt(torch.abs(circles0[...,0] - circles1[...,0]).pow(2) + torch.abs(circles0[...,1] - circles1[...,1]).pow(2))
    
    d_mask = d != 0
    d_intersected = d[d_mask].view(-1,d_mask.shape[1])
    
    circles0_intersected = circles0.expand(d_mask.shape[0],-1,-1) 
    circles1_intersected = circles1.expand(-1,d_mask.shape[1],-1) 
    shp = circles0_intersected.shape
    circles0_intersected = circles0_intersected[d_mask.unsqueeze(-1).expand_as(circles0_intersected)].view(shp)  
    circles1_intersected = circles1_intersected[d_mask.unsqueeze(-1).expand_as(circles1_intersected)].view(shp)

    rad1sqr = circles0_intersected[...,2] * circles0_intersected[...,2]
    rad2sqr = circles1_intersected[...,2] * circles1_intersected[...,2]
    ar1 = rad1sqr +  d_intersected.pow(2) - rad2sqr
    ar2 =  (2 * circles0_intersected[...,2] * d_intersected)


    angle1 = ar1 / ar2
    angle2 = (rad2sqr + (d_intersected * d_intersected) - rad1sqr) / (2 * circles1_intersected[...,2] * d_intersected)

    case_angle1_1 = angle1 < 1 
    tmp = angle1 >= -1
    case_angle1_1 = case_angle1_1 & tmp

    case_angle2_1 = angle2 < 1 
    tmp = angle2 >= -1
    case_angle2_1 = case_angle2_1 & tmp
    case_1 = case_angle2_1 & case_angle1_1


    case_angle1_2 = angle1 == 1
    case_angle1_3 = angle1 < -1
    case_angle1_4 = angle1 > 1

    case_angle2_2 = angle2 == 1
    case_angle2_3 = angle2 < -1
    case_angle2_4 = angle2 > 1
    # Check to see if the circles are overlapping
    

    theta1 = (torch.acos(angle1[case_1]) * 2)
    theta2 = (torch.acos(angle2[case_1]) * 2)

    area1 = (0.5 * theta2 * rad2sqr[case_1]) - (0.5 * rad2sqr[case_1] * torch.sin(theta2))
    area2 = (0.5 * theta1 * rad1sqr[case_1]) - (0.5 * rad1sqr[case_1] * torch.sin(theta1))

    print(torch.nonzero(d_mask)[0],case_1.shape)

    tmp_index = torch.mul((torch.nonzero(d_mask)[0] + 1).view(case_1.shape) , case_1.long())
    tmp_index_mask = tmp_index > 0
    index_tensor = tmp_index[tmp_index_mask] - 1
    result[index_tensor] = area1 + area2
    
    #Circles touch at a single degenerate point and do not intersect
    tmp_index = (torch.nonzero(d_mask).squeeze() + 1) * (case_angle1_2 & case_angle2_2).long()
    tmp_index_mask = tmp_index > 0
    index_tensor = tmp_index[tmp_index_mask] - 1
    result[index_tensor] = 0
    result[index_tensor] = 0
    
    # #Smaller circle is completely inside the larger circle. Intersecting area will be area of smaller circle
    tmp_index = torch.nonzero(d_mask).squeeze() * (case_angle1_3 & case_angle2_3).long()
    tmp_index_mask = tmp_index > 0
    index_tensor = tmp_index[tmp_index_mask] - 1
    result[index_tensor] = area_of(torch.min(circles0_intersected[index_tensor,2],circles1_intersected[index_tensor,2]))
    # #    return smallercirclearea(rad1,rad2)
    
    # #Imaginary touch points
    tmp_index = torch.nonzero(d_mask).squeeze() * (case_angle1_4 & case_angle2_4).long()
    tmp_index_mask = tmp_index > 0
    index_tensor = tmp_index[tmp_index_mask] - 1
    result[index_tensor] = 0

    #if the circle centers are the same
    result[~d_mask] = area_of(torch.min(circles0[~d_mask][:,2],circles1[~d_mask][:,2]))
    #    return smallercirclearea(rad1,rad2)
    
    return result


def assign_priors(gt_circles, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth circles and targets to priors.

    Args:
        gt_circles (num_targets, 3): ground truth circles.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 3): corner form priors
    Returns:
        circles (num_priors, 3): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_circles.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    circles = gt_circles[best_target_per_prior_index]
    return circles, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def hard_nms(circle_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        circle_scores (N, 4): circles and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept circles
    """
    scores = circle_scores[:, -1]
    circles = circle_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = circles[current, :]
        indexes = indexes[1:]
        rest_circles = circles[indexes, :]
        iou = iou_of(
            rest_circles,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return circle_scores[picked, :]


def nms(circle_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(circle_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(circle_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(circle_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        circle_scores (N, 4): circles in corner-form and probabilities.
        score_threshold: circles with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_circle_scores (K, 4): results of NMS.
    """
    picked_circle_scores = []
    while circle_scores.size(0) > 0:
        max_score_index = torch.argmax(circle_scores[:, 3])
        cur_circle_prob = torch.tensor(circle_scores[max_score_index, :])
        picked_circle_scores.append(cur_circle_prob)
        if len(picked_circle_scores) == top_k > 0 or circle_scores.size(0) == 1:
            break
        cur_circle = cur_circle_prob[:-1]
        circle_scores[max_score_index, :] = circle_scores[-1, :]
        circle_scores = circle_scores[:-1, :]
        ious = iou_of(cur_circle.unsqueeze(0), circle_scores[:, :-1])
        circle_scores[:, -1] = circle_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        circle_scores = circle_scores[circle_scores[:, -1] > score_threshold, :]
    if len(picked_circle_scores) > 0:
        return torch.stack(picked_circle_scores)
    else:
        return torch.tensor([])



