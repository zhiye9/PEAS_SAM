import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import supervision as sv
import pandas as pd
import pickle
import os

with open("/home/zhi/data/PEAS/processed_data/img_id", "rb") as fp:
    img_id = pickle.load(fp)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

e = []
num = len(img_id)
for i in range(num):
    image = cv2.cvtColor(cv2.imread(img_id[i]), cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    e.append(np.squeeze(predictor.get_image_embedding().cpu().numpy()))
    print("\r Process{}%".format(round((i+1)*100/num)), end="")

with open("/home/zhi/data/PEAS/processed_data/image_embedding_vit_h", "wb") as fp:   
    pickle.dump(e, fp)
with open("/home/zhi/data/PEAS/processed_data/image_embedding_vit_h", "rb") as fp:
    e = pickle.load(fp)