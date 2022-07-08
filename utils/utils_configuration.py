import os
import numpy as np


VOC_CLASS_NAMES = (
    open(os.path.join(os.getcwd(), "VOCdevkit", "VOC_CLASSES.txt"))
    .read()
    .lower()
    .strip()
    .split("\n")
)
INDEX_TO_NAMES = dict(zip(range(len(VOC_CLASS_NAMES)), VOC_CLASS_NAMES))

imagenet_rgb_means = np.array([123, 117, 104], dtype=np.float32)
voc_ssd300_configuration = {
    "num_classes": 21,
    "feature_map_sizes": [38, 19, 10, 5, 3, 1],
    "image_size": 300,
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "positive_negative_iou_threshold": 0.5,
    "negative_positive_ration": 3,
}

train_process_configuration = {
    "init_batchsize": 32,
    "unfreeze_batchsize": 16,
    "resume_train": False,
    "resume_weights_path": None,
    "warmup_epoch": 5,
    "init_epoch": 0,
    "unfreeze_epoch": 40,
    "end_epoch": 120,
    "num_workers": 4,
    "use_cuda": False,
    "init_lr": 1e-3,
    "unfreeze_lr": 1e-4,
    "min_lr": 1e-5,
    "save_weights_folder": "weights/",
    "num_freeze_layers": 28,
    "save_period": 10,
}

predict_process_configuration = {
    "index_to_labels": INDEX_TO_NAMES,
    "image_size": 300,
    "font_file_path": "C:/Windows/Fonts/simhei.ttf",
    "variance": [0.1, 0.2],
    "num_classes": 21,
    "nms_iou_threshold": 0.5,
    "confidence_score": 0.4,
    "img_file_path": "./imgs/cat.jpg",
    "trained_weights_path": "./weights/gpu_trained_converge_voc2007_weights.pth",
}
