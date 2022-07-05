from torch.utils.data import Dataset
from utils_augmentation import BaseTransform, MultipleTransform
import xml.etree.ElementTree as ET
import numpy as np
import torch
import cv2
import os


class VOCDataset(Dataset):
    def __init__(self, mode, size=300, keep_difficult=False):
        """
        Args:
            mode(string): collection of data sources(support 'train'/'val'/'test')
            size(int): the input image size
            keep_difficult(bool): check the diificult bounding box
        """
        self.mode = mode
        self.size = size
        self.keep_difficult = keep_difficult

        self.augment = (
            MultipleTransform(self.size)
            if self.mode == "train"
            else BaseTransform(self.size)
        )
        self.VOCDATASET_PATH = os.path.join("..", "VOCdevkit")
        self.voc_class_names = (
            open(os.path.join(self.VOCDATASET_PATH, "VOC_CLASSES.txt"))
            .read()
            .lower()
            .strip()
            .split("\n")
        )
        self.voc_file_names = (
            open(
                os.path.join(
                    self.VOCDATASET_PATH,
                    "VOC2007",
                    "ImageSets",
                    "Main",
                    f"{mode}.txt",
                )
            )
            .read()
            .lower()
            .strip()
            .split("\n")
        )
        self.class_names_to_logits_index = dict(
            zip(self.voc_class_names, range(len(self.voc_class_names)))
        )
        self.logits_index_to_class_names = dict(
            zip(range(len(self.voc_class_names)), self.voc_class_names)
        )

    def __getitem__(self, index):
        voc_file_name = self.voc_file_names[index]
        one_raw_img = self.readImage(voc_file_name + ".jpg")
        (
            _,
            _,
            _,
            bounding_boxes_relative_coords,
            bounding_boxes_class_labels,
        ) = self.readAnnotation(voc_file_name + ".xml")
        (
            one_transformed_img,
            bounding_boxes_relative_coords,
            bounding_boxes_class_labels,
        ) = self.augment(
            one_raw_img,
            bounding_boxes_relative_coords,
            bounding_boxes_class_labels,
        )
        ground_truth = np.concatenate(
            [bounding_boxes_relative_coords, bounding_boxes_class_labels],
            axis=1,
        )
        return (
            torch.from_numpy(one_transformed_img).permute(2, 0, 1).type(torch.float32),
            torch.from_numpy(ground_truth).type(torch.float32),
        )

    def __len__(self):
        return len(self.voc_file_names)

    def readAnnotation(self, xml_file_name):
        """
        Args:
            xml_file_name(string):  "000001.xml"
        Return:
            1. the height of the input image(int)
            2. the width of the input image(int)
            3. the absolute coords of the bounding boxes in the input image(ndarray)
               e.g. [[ 48 240 195 371],[ 8 12 352 498]]
                    shape: [num_bounding_boxes,4]
            4. the relative coords of the bounding boxes in the input image(ndarray)
               e.g. [[0.13 0.48 0.55 0.74],[0.02 0.02 0.99 0.99]]
                    shape: [num_bounding_boxes,4]
            5. the class logit labels for each bounding box in the input image(ndarray)
               e.g. [[11],[14]]
                    shape: [num_bounding_boxes,1]
        """
        target = ET.parse(
            os.path.join(self.VOCDATASET_PATH, "VOC2007", "Annotations", xml_file_name)
        ).getroot()
        bounding_boxes_absolute_coords = []
        bounding_boxes_relative_coords = []
        bounding_boxes_class_labels = []

        img_size = target.find("size")
        img_height = int(img_size.find("height").text)
        img_width = int(img_size.find("width").text)

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue

            name = obj.find("name").text.lower().strip()
            class_label = self.class_names_to_logits_index[name]
            bounding_boxes_class_labels.append([class_label])

            one_bounding_box = obj.find("bndbox")
            bounding_boxes_absolute_coords.append(
                [
                    int(one_bounding_box.find("xmin").text),
                    int(one_bounding_box.find("ymin").text),
                    int(one_bounding_box.find("xmax").text),
                    int(one_bounding_box.find("ymax").text),
                ]
            )
            bounding_boxes_relative_coords.append(
                [
                    int(one_bounding_box.find("xmin").text) / img_width,
                    int(one_bounding_box.find("ymin").text) / img_height,
                    int(one_bounding_box.find("xmax").text) / img_width,
                    int(one_bounding_box.find("ymax").text) / img_height,
                ]
            )
        return (
            img_height,
            img_width,
            np.array(bounding_boxes_absolute_coords),
            np.array(bounding_boxes_relative_coords),
            np.array(bounding_boxes_class_labels),
        )

    def readImage(self, img_name):
        img = cv2.imread(
            os.path.join(self.VOCDATASET_PATH, "VOC2007", "JPEGImages", img_name)
        )
        img = img[:, :, (2, 1, 0)]
        return img
