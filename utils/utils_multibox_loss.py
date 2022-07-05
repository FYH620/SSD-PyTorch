import torch
import torch.nn.functional as F
from torch import nn
from utils_anchor import AnchorBoxes
from utils_auxiliary import (
    iouMatrix,
    centerCoordsToPointCoords,
    encodeCenterCoords,
    pointCoordsToCenterCoords,
)


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, overlap_threshold, neg_pos_ratio, use_cuda):
        super().__init__()
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.use_cuda = use_cuda
        self.anchor_boxes_coords = AnchorBoxes().getAnchorBoxesForEachImage()
        self.loc_variance = [0.1, 0.2]

    def forward(self, loc_data, conf_data, targets):
        num_batch = loc_data.size(0)
        anchor_boxes = self.anchor_boxes.copy()
        num_anchors = anchor_boxes.size(0)
        loc_t = torch.tensor(num_batch, num_anchors, 4)
        conf_t = torch.LongTensor(num_batch, num_anchors, self.num_classes)
        for index in range(len(num_batch)):
            loc_t[index], conf_t[index] = self._match(targets[index].data)

        if self.use_cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

    def _match(self, target):
        target_point_coords = target[:, :-1]
        target_class_labels = target[:, -1]
        iou_matrix = iouMatrix(
            target_point_coords,
            centerCoordsToPointCoords(self.anchor_boxes_coords),
        )
        # the anchors with which ground truth has the largest iou
        anchor_overlap, anchor_index = torch.max(iou_matrix, dim=1)
        # the ground truths with which anchors has the largest iou
        gt_overlap, gt_index = torch.max(iou_matrix, dim=0)
        gt_overlap.index_fill_(dim=0, index=anchor_index, value=1)
        for index in range(len(anchor_index)):
            gt_index[anchor_index[index]] = index

        gt_coords = target_point_coords[gt_index]
        loc = encodeCenterCoords(
            self.anchor_boxes_coords,
            pointCoordsToCenterCoords(gt_coords),
            self.loc_variance,
        )

        conf = target_class_labels[gt_index] + 1
        conf[gt_overlap < self.overlap_threshold] = 0
        return loc, conf
