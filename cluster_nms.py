import torch
import time

def intersect(box_a, box_b):
    """Compute the intersection of two sets of boxes. The intersection of two boxes is a box defined by the intersection of the x and y ranges of the two boxes."""
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b, iscrowd: bool=False):
    """Compute the Jaccard overlap (IoU) of two sets of boxes."""
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

class NMSMethods:
    def __init__(self, conf_thresh):
        self.conf_thresh = conf_thresh

    def fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200, second_threshold: bool = False):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes * num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes * num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        keep = (iou_max <= iou_threshold)
        if second_threshold:
            keep *= (scores > self.conf_thresh)
        keep *= (scores > 0.01)

        classes = torch.arange(num_classes, device=boxes.device).repeat_interleave(num_dets)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores.view(-1)[keep]

        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def cluster_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200, second_threshold: bool = False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes * num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes * num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A = B
            maxA, _ = A.max(dim=1)
            E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
            B = iou.mul(E)
            if A.equal(B):
                break
        keep = (maxA <= iou_threshold)
        keep *= (scores.view(-1) > 0.01)

        classes = torch.arange(num_classes, device=boxes.device).repeat_interleave(num_dets)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores.view(-1)[keep]

        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores
