import cv2
import numpy as np
import torch
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SegmentWithSAM:
    def __init__(self, checkpoint='sam_vit_b_01ec64.pth', model_type='vit_b'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7
        )

    def get_masks(self, image):
        masks = self.mask_generator.generate(image)
        sorted_masks = sorted(masks, key=lambda m: np.sum(
            m['segmentation']), reverse=True)
        return sorted_masks

    def plot_masks(self, image, masks):
        annotated = image.copy()

        for mask in masks:
            m = mask['segmentation']

            color = np.random.randint(0, 255, (3,), dtype=np.uint8)

            annotated[m] = annotated[m] * 0.5 + color * 0.5

            contour = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(annotated, contour, -1, (0, 255, 0), 2)

        return annotated
