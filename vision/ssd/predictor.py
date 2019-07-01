import torch
from torch.utils.data import DataLoader, TensorDataset
from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict_pieces_mod(self, pieces, offsets, top_k=-1, prob_threshold=None):
        BATCH_SIZE = 10
        cpu_device = torch.device("cpu")
        height, width, _ = pieces[0].shape
        images = []
        for image in pieces:
            image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        images = images.to(self.device)
        all_scores = []
        all_boxes = []
        self.timer.start()

        for idx,batch in enumerate(DataLoader(TensorDataset(images),batch_size=BATCH_SIZE)):
            with torch.no_grad():
                self.timer.start()
                scores, boxes = self.net.forward(batch[0])
                boxes *= 300

                for idx2,offset in enumerate(offsets[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]):
                    boxes[idx2,:, 0] += offset[0]
                    boxes[idx2,:, 1] +=  offset[1]
                    boxes[idx2,:, 2] += offset[0]
                    boxes[idx2,:, 3] +=  offset[1]
                boxes[:,:,0] /= 3300
                boxes[:,:,1] /= 2700
                boxes[:,:,2] /= 3300
                boxes[:,:,3] /= 2700
                all_scores.append(scores)
                all_boxes.append(boxes)
        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        all_boxes = torch.reshape(all_boxes,(-1,4))
        all_scores = torch.reshape(all_scores,(-1,2))
      
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = all_boxes#.to(cpu_device)
        scores = all_scores#.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)

            box_probs = box_utils.nms(box_probs, self.nms_method,
                                    score_threshold=prob_threshold,
                                    iou_threshold=self.iou_threshold,
                                    sigma=self.sigma,
                                    top_k=top_k,
                                    candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= 3300
        picked_box_probs[:, 1] *= 2700
        picked_box_probs[:, 2] *= 3300
        picked_box_probs[:, 3] *= 2700
        picked_box_probs = picked_box_probs.to(cpu_device)
        print("Inference time: ", self.timer.end())

        return picked_box_probs[:, :4],picked_labels,picked_box_probs[:, 4]
 
    def predict_pieces(self, pieces, offsets, top_k=-1, prob_threshold=None):
        BATCH_SIZE = 10
        cpu_device = torch.device("cpu")
        height, width, _ = pieces[0].shape
        images = []
        for image in pieces:
            image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        images = images.to(self.device)
        all_scores = []
        all_boxes = []
        for batch in DataLoader(TensorDataset(images),batch_size=BATCH_SIZE):
            with torch.no_grad():
                self.timer.start()
                scores, boxes = self.net.forward(batch[0])
                print("Inference time: ", self.timer.end())
                all_scores.append(scores)
                all_boxes.append(boxes)
        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        
        result_box = []
        labels = []
        result_probs = []
        for boxes,scores,offset in zip(all_boxes,all_scores,offsets):
            
            if not prob_threshold:
                prob_threshold = self.filter_threshold
            # this version of nms is slower on GPU, so we move data to CPU.
            boxes = boxes.to(cpu_device)
            scores = scores.to(cpu_device)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                mask = probs > prob_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                subset_boxes = boxes[mask, :]
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
                
                box_probs = box_utils.nms(box_probs, self.nms_method,
                                        score_threshold=prob_threshold,
                                        iou_threshold=self.iou_threshold,
                                        sigma=self.sigma,
                                        top_k=top_k,
                                        candidate_size=self.candidate_size)
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
            if not picked_box_probs:
                # result_box.append(torch.tensor([]))
                # labels.append(torch.tensor([]))
                # result_probs.append(torch.tensor([]))
                continue
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
            picked_box_probs[:, 0] += offset[0]
            picked_box_probs[:, 1] +=  offset[1]
            picked_box_probs[:, 2] += offset[0]
            picked_box_probs[:, 3] +=  offset[1]

            result_box.append(picked_box_probs[:, :4])
            labels.append(torch.tensor(picked_labels))
            result_probs.append(picked_box_probs[:, 4])
        return torch.cat(result_box),torch.cat(labels),torch.cat(result_probs)

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]