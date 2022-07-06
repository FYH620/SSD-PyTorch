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

    def forward(self, predict_offset_coords, predict_confidences, targets):
        num_batch = len(predict_offset_coords)
        anchor_boxes = self.anchor_boxes.copy()
        num_anchors = len(anchor_boxes)
        offset_coords_labels = torch.tensor(num_batch, num_anchors, 4)
        confidence_labels = torch.LongTensor(num_batch, num_anchors)
        for index in range(len(num_batch)):
            offset_coords_labels[index], confidence_labels[index] = self._match(
                targets[index].data
            )

        if self.use_cuda:
            offset_coords_labels = offset_coords_labels.cuda()
            confidence_labels = confidence_labels.cuda()

        is_positive_anchors = confidence_labels > 0
        positive_coords_labels = offset_coords_labels[
            is_positive_anchors.unsqueeze(dim=is_positive_anchors.dim()).expand_as(
                offset_coords_labels
            )
        ].view(-1, 4)
        predict_positive_coords = predict_offset_coords[is_positive_anchors].view(-1, 4)
        loss_location = F.smooth_l1_loss(
            predict_positive_coords,
            positive_coords_labels,
            size_average=False,
        )

        num_positive = is_positive_anchors.sum(dim=-1, keepdim=True)
        num_total_positive = num_positive.data.sum()
        num_negative = torch.clamp(
            self.neg_pos_ratio * num_positive,
            max=num_anchors - 1,
        )
        loss_classification_auxiliary = self._negativeLogSoftmaxClassficationLoss(
            predict_confidences.view(-1, self.num_classes),
            confidence_labels.view(-1, 1),
        ).view(num_batch, -1)
        loss_classification_auxiliary[is_positive_anchors] = 0
        _, loss_classification_auxiliary_rank = loss_classification_auxiliary.sort(
            dim=1,
            descending=True,
        )
        _, index_rank = loss_classification_auxiliary_rank.sort(dim=1)
        neg_index = index_rank < (num_negative.expand_as(index_rank))
        neg_mask = neg_index.unsqueeze(neg_index.dim()).expand_as(predict_confidences)
        pos_mask = is_positive_anchors.unsqueeze(is_positive_anchors.dim()).expand_as(
            predict_confidences
        )
        predict_c = predict_confidences[(pos_mask + neg_mask).gt(0)].view(
            -1, self.num_classes
        )
        label_c = confidence_labels[(is_positive_anchors + neg_index).gt(0)]
        loss_classification = F.cross_entropy(predict_c, label_c, size_average=False)
        loss_location /= num_total_positive
        loss_classification /= num_total_positive
        return loss_location, loss_classification

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

    def _safeLogSumExpFunction(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)
        return x_max + (x - x_max).exp().sum(dim=1, keepdim=True).log()

    def _negativeLogSoftmaxClassficationLoss(self, x, gt_class_labels):
        return self._safeLogSumExpFunction(x) - torch.gather(
            x,
            dim=1,
            index=gt_class_labels,
        )
