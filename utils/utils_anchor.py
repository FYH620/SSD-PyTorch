import torch
from math import sqrt
from itertools import product


class AnchorBoxes:
    def __init__(self):
        self.image_size = 300
        self.feature_map_sizes = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def getAnchorBoxesForEachImage(self):
        anchor_boxes = []
        for step, feature_map_size in enumerate(self.feature_map_sizes):
            for i, j in product(range(feature_map_size), repeat=2):
                center_relative_x = (j + 0.5) / feature_map_size
                center_relative_y = (i + 0.5) / feature_map_size
                small_anchor_scale = self.min_sizes[step] / self.image_size
                anchor_boxes += [
                    center_relative_x,
                    center_relative_y,
                    small_anchor_scale,
                    small_anchor_scale,
                ]
                large_anchor_scale = sqrt(
                    small_anchor_scale * (self.max_sizes[step] / self.image_size)
                )
                anchor_boxes += [
                    center_relative_x,
                    center_relative_y,
                    large_anchor_scale,
                    large_anchor_scale,
                ]

                for aspect_ratio in self.aspect_ratios[step]:
                    anchor_boxes += [
                        center_relative_x,
                        center_relative_y,
                        small_anchor_scale * sqrt(aspect_ratio),
                        small_anchor_scale / sqrt(aspect_ratio),
                    ]
                    anchor_boxes += [
                        center_relative_x,
                        center_relative_y,
                        small_anchor_scale / sqrt(aspect_ratio),
                        small_anchor_scale * sqrt(aspect_ratio),
                    ]
        final_anchor_boxes = torch.tensor(anchor_boxes).view(-1, 4)
        return torch.clamp(final_anchor_boxes, min=0, max=1)
