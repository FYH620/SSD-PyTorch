import numpy as np


imagenet_rgb_means = np.array([123, 117, 104], dtype=np.float32)
voc_ssd300_configuration = {
    "num_classes": 21,
    "feature_map_sizes": [38, 19, 10, 5, 3, 1],
    "image_size": 300,
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "save_period": 10,
}
