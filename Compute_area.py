import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import supervision as sv
import pandas as pd
import pickle
import os
import re
import plotly.express as px
import time
from scipy import ndimage
from skimage.measure import label, regionprops
import math
import json

with open("/home/zhi/data/PEAS/processed_data/img_id", "rb") as fp:
    img_id = pickle.load(fp)

def remove_corner(img_path):
    fill_color = (255, 255, 255)
    img = cv2.imread(img_path)

    # Define the coordinates for the rectangular bounding boxes (x, y, width, height)
    bounding_boxes = [
        (0, 0, 200, 100),  # Example: Top-left corner
        (0, img.shape[0] - 95, 295, 95),  # Example: Bottom-left corner
        (img.shape[1] - 320, img.shape[0]-100, 320, 100),  # Example: Bottom-right corner
    ]

    for box in bounding_boxes:
        x, y, width, height = box
        img[y:y + height, x:x + width] = fill_color

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_substring(input_string, NGB = True):
    # Use regular expressions to find the desired substring
    match = re.search(r'NGB(.*?)_', input_string)
    
    if match:
        if NGB:
            return "NGB" + match.group(1)
        else:
            return match.group(1)
    else:
        return None

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

#mask_idx = []
#mask_areas = []
#mask_eccentricity = []
#mask_rgb = []
#mask_solidity = []
#mask_major_length = []
#mask_minor_length = []
#mask_roundness = []
#mask_features = []

num = len(img_id)
dict_features = {}

for i in range(num):
    start_time = time.time()
    image = remove_corner(img_id[i])
    dict_id = img_id[i].split("/")[6] + '/' + img_id[i].split("/")[7]
    id1 = extract_substring(img_id[i])
    id2 = extract_substring(img_id[i], NGB = False)
    mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh=0.2, min_mask_region_area=600)
    mask_sorted = sorted(mask_generator.generate(image), key=lambda x: x['area'], reverse=True)
    areas = []
    eccentricity = []
    rgb = []
    solidity = []
    perimeter = []
    idx = []
    major_length = []
    minor_length = []
    roundness =[]
    for k in range(len(mask_sorted)):
        single_mask = mask_sorted[k]
        if (single_mask['area'] < 25000) & (single_mask['area'] > 2000):
            m = single_mask['segmentation'].astype(np.uint8)
            regions = regionprops(m)
            if len(regions) == 1:
                res = cv2.bitwise_and(image, image, mask=m)
                indict = 0
                for j in range(3):
                    num_unique = np.unique(res[:, :, j], return_counts= True)
                    if (255 in num_unique[0]) & (len(num_unique[0]) == 2):
                        indict += 1
                    elif (255 in num_unique[0]) & (num_unique[1][-1] == sorted(num_unique[1], reverse = True)[1]) & (num_unique[1][-1] > sorted(num_unique[1], reverse = True)[2]*50):
                        indict += 1
                if indict != 3:
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        # Get the bounding box of the contour
                        x, y, w, h = cv2.boundingRect(contour)
                        height, width = image.shape[:2]
                        # Check if the bounding box is entirely within the image
                        if (x > 0 and y > 0 and x + w < width and y + h < height):
                            if(x < 200 and y > 100 and y + h < height - 95) or (x < 295 and x > 200 and y + h < height - 95) or (x > 295 and x + w < width - 320) or (x + w > width - 320 and y + h < height - 100):
                                idx.append(k)
                                areas.append(regions[0]['area'])
                                eccentricity.append(regions[0].eccentricity)
                                rgb.append(res[m > 0])
                                solidity.append(regions[0]['solidity'])
                                perimeter.append(regions[0]['perimeter'])
                                major_length.append(regions[0]['axis_major_length'])
                                minor_length.append(regions[0]['axis_minor_length'])
                                roundness.append((4*math.pi*regions[0]['area'])/(regions[0]['perimeter']**2))

    dict_mask_features = {'id1': id1, 'id2': id2, 'msk_idx': idx, 'areas': areas, 'eccentricity': eccentricity, 'rgb_value': rgb,'solidity': solidity, 'perimeter': perimeter, 'major_length': major_length, 'minor_length': minor_length, 'roundness': roundness}
    dict_features[dict_id] = dict_mask_features
    
    #mask_features.append({'id1': id1, 'id2': id2, 'msk_idx': idx, 'areas': areas, 'eccentricity': eccentricity, 'rgb_value': rgb,'solidity': solidity, 'perimeter': perimeter, 'major_length': major_length, 'minor_length': minor_length, 'roundness': roundness})
    #mask_idx.append(idx)
    #mask_areas.append(areas)
    #mask_eccentricity.append(eccentricity)
    #mask_rgb.append(rgb)
    #mask_solidity.append(solidity)
    #mask_perimeter.append(perimeter)
    #mask_major_length.append(major_length)
    #mask_minor_length.append(minor_length)
    #mask_roundness.append(roundness)

    #print(time.time() - start_time)
    print("\r Process{}%".format(round((i+1)*100/num)), end="")
    
with open("/home/zhi/data/PEAS/processed_data/dict_feature", "wb") as fp:   
    pickle.dump(dict_features, fp)

#list_msk = [mask_idx, mask_areas, mask_eccentricity, mask_rgb, mask_solidity, mask_perimeter, mask_major_length, mask_minor_length, mask_roundness]
#with open("/home/zhi/data/PEAS/processed_data/list_msk", "wb") as fp:   
#    pickle.dump(list_msk, fp)
    
#with open("/home/zhi/data/PEAS/processed_data/dict_msk", "wb") as fp:   
#    pickle.dump(mask_features, fp)

