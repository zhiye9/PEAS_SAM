import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import supervision as sv

image = cv2.imread('/home/zhi/data/PEAS/images/NGB100006_1_2_SD_seed_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
model=sam,
points_per_side=32,
pred_iou_thresh=0.86,
stability_score_thresh=0.92,
crop_n_layers=1,
crop_n_points_downscale_factor=2,
min_mask_region_area=100, # Requires open-cv to run post-processing
)
masks = mask_generator.generate(image)

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image, detections)

predictor = SamPredictor(sam)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding()
plt.figure(figsize=(20,20))
plt.imshow(image_embedding)
plt.axis('off')
plt.show()

--------------------------------------------------------------------------
img_path = "/home/zhi/nas/PEAS/PEAS_images"


