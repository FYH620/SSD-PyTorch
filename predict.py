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
from utils.utils_dataload import VOCDataset


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

            box_index, box_index_length = nonMaximumSuppression(
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

    def drawBoundingBoxesOnImage(self, result_dict):
        img = cv2.imread(self.img_file_path)
        for k, v in result_dict.items():
            v = v.detach().numpy().astype(np.int32)
            for i in range(len(v)):
                min_xy = (v[i, 0], v[i, 1])
                max_xy = (v[i, 2], v[i, 3])
                cv2.rectangle(
                    img,
                    min_xy,
                    max_xy,
                    color=(241, 86, 66),
                    thickness=2,
                )
                texts = VOCDataset("test").logits_index_to_class_names[k]
                cv2.putText(
                    img,
                    texts,
                    min_xy,
                    cv2.FONT_HERSHEY_PLAIN,
                    color=(16, 213, 243),
                    thickness=1,
                    fontScale=3,
                )

        cv2.imshow("img-with-box:press q to exit", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
