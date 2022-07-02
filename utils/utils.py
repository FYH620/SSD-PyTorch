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
    inter = torch.clamp((max_xy, min_xy), min=0)
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
