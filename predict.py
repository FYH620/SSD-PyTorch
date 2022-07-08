import cv2
import torch
import numpy as np
from nets.ssd import SSD
from utils.utils_auxiliary import (
    decodeCenterCoords,
    nonMaximumSuppression,
    centerCoordsToPointCoords,
)
from utils.utils_anchor import AnchorBoxes
from utils.utils_augmentation import BaseTransform
from utils.utils_configuration import predict_process_configuration as config
from torch import Tensor
from colorsys import hsv_to_rgb
from PIL import ImageDraw, ImageFont, Image


class ObjectDetection:
    def __init__(
        self,
        variance,
        num_classes,
        nms_iou_threshold,
        confidence_threshold,
        img_file_path,
        trained_weights_path,
    ):
        self.variance = variance
        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.confidence_threshold = confidence_threshold
        self.trained_weights_path = trained_weights_path
        self.img_file_path = img_file_path
        self.ssd_model = SSD(mode="test")
        self.defined_anchor_coords = AnchorBoxes().getAnchorBoxesForEachImage()

    def getBoundingBoxesResult(self):
        result_dict = dict()
        self.ssd_model.loadTrainedWeights(self.trained_weights_path)
        self.ssd_model.eval()

        raw_predict_img = cv2.imread(self.img_file_path)
        img_h, img_w, _ = raw_predict_img.shape
        predict_img, _, _ = BaseTransform(size=300)(raw_predict_img, None, None)
        predict_img = torch.from_numpy(predict_img).permute(2, 0, 1).unsqueeze(0)

        predict_class_confidences, predict_offset_coords = self.ssd_model(predict_img)
        predict_class_confidences = predict_class_confidences.squeeze(0).permute(1, 0)
        predict_offset_coords.squeeze_(0)

        ground_truth_center_coords = decodeCenterCoords(
            self.defined_anchor_coords, predict_offset_coords, self.variance
        )
        ground_truth_coords = centerCoordsToPointCoords(ground_truth_center_coords)

        for positive_class_label in range(1, self.num_classes):
            class_mask = predict_class_confidences[positive_class_label].gt(
                self.confidence_threshold
            )
            location_mask = class_mask.unsqueeze(1).expand_as(ground_truth_coords)
            class_scores = predict_class_confidences[positive_class_label][class_mask]
            if len(class_scores) == 0:
                continue
            coords = ground_truth_coords[location_mask].view(-1, 4)

            box_index, _ = nonMaximumSuppression(
                coords, class_scores, self.nms_iou_threshold
            )

            final_coords = coords[box_index]
            final_coords[:, [0, 2]] *= img_w
            final_coords[:, [1, 3]] *= img_h

            result_dict[positive_class_label - 1] = torch.cat(
                [final_coords.type(torch.int32), class_scores[box_index].unsqueeze(1)],
                dim=1,
            )

        return result_dict

    def drawOnePictureWithBox(
        self,
        index_to_labels,
        box_coords,
        box_labels,
        box_scores=None,
        resize_scale=800,
        is_use_scores=True,
        is_ralative_coords=False,
    ):

        img = cv2.imread(self.img_file_path)
        if img.shape[2] != 3:
            raise ValueError("We only support RGB images.")

        if isinstance(box_coords, Tensor):
            box_coords = box_coords.data.numpy()
        if isinstance(box_labels, Tensor):
            box_labels = box_labels.data.numpy()
        if isinstance(box_scores, Tensor):
            box_scores = box_scores.data.numpy()
        box_coords = np.array(box_coords)
        box_labels = np.array(box_labels)
        if is_use_scores:
            box_scores = np.array(box_scores)

        img_height, img_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        font = ImageFont.truetype(
            font=config["font_file_path"],
            size=np.floor(3e-2 * img_width + 0.5).astype("int32"),
        )
        thickness = max((img_height + img_width) // resize_scale, 1)
        num_classes = len(index_to_labels.keys())
        hsv_tuples = [(x / num_classes, 1, 1) for x in range(num_classes)]
        colors = list(map(lambda x: hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(
                lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors,
            )
        )
        if is_ralative_coords:
            box_coords[:, [0, 2]] *= img_width
            box_coords[:, [1, 3]] *= img_height
        box_coords = box_coords.astype(np.int32)
        box_labels = box_labels.astype(np.int32)
        if is_use_scores:
            box_scores = box_scores.astype(np.float32)

        for i, c in enumerate(box_labels):
            class_name = index_to_labels[c]
            box_coord = box_coords[i]
            if is_use_scores:
                box_score = box_scores[i]

            left, top, right, bottom = box_coord
            left = max(0, np.floor(left))
            top = max(0, np.floor(top))
            right = min(img_width, np.floor(right))
            bottom = min(img_height, np.floor(bottom))

            if is_use_scores:
                label = "{} {:.2f}".format(class_name, box_score)
            else:
                label = "{}".format(class_name)
            draw = ImageDraw.Draw(img)
            label_size = draw.textsize(label, font)
            print(left, top, right, bottom, label)
            label = label.encode("utf-8")

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c],
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c],
            )
            draw.text(text_origin, str(label, "UTF-8"), fill=(0, 0, 0), font=font)
            del draw

        img.show()


def predictStaticImage():
    detector = ObjectDetection(
        variance=config["variance"],
        num_classes=config["num_classes"],
        nms_iou_threshold=config["nms_iou_threshold"],
        confidence_threshold=config["confidence_score"],
        img_file_path=config["img_file_path"],
        trained_weights_path=config["trained_weights_path"],
    )
    result_dict = detector.getBoundingBoxesResult()

    box_coords, box_labels, box_scores = [], [], []
    for class_label, boxes_coords in result_dict.items():
        for one_box in boxes_coords:
            box_coords.append(one_box[:4].data.numpy())
            box_labels.append(class_label)
            box_scores.append(one_box[4].data.numpy())

    detector.drawOnePictureWithBox(
        index_to_labels=config["index_to_labels"],
        box_coords=box_coords,
        box_labels=box_labels,
        box_scores=box_scores,
        resize_scale=config["image_size"],
        is_use_scores=True,
        is_ralative_coords=False,
    )


if __name__ == "__main__":
    predictStaticImage()
