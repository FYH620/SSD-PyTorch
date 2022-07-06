import torch
from torch import Tensor


def intersactArea(boxes_a: Tensor, boxes_b: Tensor):
    A = boxes_a.size(0)
    B = boxes_b.size(0)
    min_xy = torch.max(
        boxes_a[:, :2].unsqueeze(dim=1).expand(A, B, 2),
        boxes_b[:, :2].unsqueeze(dim=0).expand(A, B, 2),
    )
    max_xy = torch.min(
        boxes_a[:, 2:].unsqueeze(dim=1).expand(A, B, 2),
        boxes_b[:, 2:].unsqueeze(dim=0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def iouMatrix(boxes_a: Tensor, boxes_b: Tensor):
    A = boxes_a.size(0)
    B = boxes_b.size(0)
    interact_areas = intersactArea(boxes_a, boxes_b)
    areas_a = (
        ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
        .unsqueeze(dim=1)
        .expand(A, B)
    )
    areas_b = (
        ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))
        .unsqueeze(dim=0)
        .expand(A, B)
    )
    return interact_areas / (areas_a + areas_b - interact_areas)


def centerCoordsToPointCoords(center_coords: Tensor):
    point_coords = torch.zeros_like(center_coords)
    point_coords[:, 0] = center_coords[:, 0] - (center_coords[:, 2] / 2)
    point_coords[:, 1] = center_coords[:, 1] - (center_coords[:, 3] / 2)
    point_coords[:, 2] = center_coords[:, 0] + (center_coords[:, 2] / 2)
    point_coords[:, 3] = center_coords[:, 1] + (center_coords[:, 3] / 2)
    return torch.clamp(point_coords, min=0, max=1)


def pointCoordsToCenterCoords(point_coords: Tensor):
    center_coords = torch.zeros_like(point_coords)
    center_coords[:, 0] = (point_coords[:, 0] + point_coords[:, 2]) / 2
    center_coords[:, 1] = (point_coords[:, 1] + point_coords[:, 3]) / 2
    center_coords[:, 2] = point_coords[:, 2] - point_coords[:, 0]
    center_coords[:, 3] = point_coords[:, 3] - point_coords[:, 1]
    return torch.clamp(center_coords, min=0, max=1)


def encodeCenterCoords(anchor_coords, gt_coords, variance):
    offset_coords = torch.zeros_like(anchor_coords)
    offset_coords[:, 0] = (gt_coords[:, 0] - anchor_coords[:, 0]) / (
        variance[0] * anchor_coords[:, 2]
    )
    offset_coords[:, 1] = (gt_coords[:, 1] - anchor_coords[:, 1]) / (
        variance[0] * anchor_coords[:, 3]
    )
    offset_coords[:, 2] = torch.log(gt_coords[:, 2] / anchor_coords[:, 2]) / variance[1]
    offset_coords[:, 3] = torch.log(gt_coords[:, 3] / anchor_coords[:, 3]) / variance[1]
    return offset_coords


def decodeCenterCoords(anchor_coords, offset_coords, variance):
    gt_coords = torch.zeros_like(anchor_coords)
    gt_coords[:, 0] = (
        anchor_coords[:, 2] * offset_coords[:, 0] * variance[0] + anchor_coords[:, 0]
    )
    gt_coords[:, 1] = (
        anchor_coords[:, 3] * offset_coords[:, 1] * variance[0] + anchor_coords[:, 1]
    )
    gt_coords[:, 2] = anchor_coords[:, 2] * torch.exp(offset_coords[:, 2] * variance[1])
    gt_coords[:, 3] = anchor_coords[:, 3] * torch.exp(offset_coords[:, 3] * variance[1])
    return gt_coords
