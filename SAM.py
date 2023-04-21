import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import supervision as sv
import pandas as pd
import pickle
import os
from sklearn.manifold import TSNE
import plotly.express as px

img_temp =  '/home/zhi/nas/PEAS/PEAS_images/OneDrive_6_23-03-2023/NGB106117_1_8_SSD_seed_1.png'
image = cv2.imread(img_temp)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

mask_generator = SamAutomaticMaskGenerator(sam)

#mask_generator = SamAutomaticMaskGenerator(
#model=sam,
#points_per_side=32,
#pred_iou_thresh=0.86,
#stability_score_thresh=0.92,
#crop_n_layers=1,
#crop_n_points_downscale_factor=2,
#min_mask_region_area=100, # Requires open-cv to run post-processing
#)

masks = mask_generator.generate(image)

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image, detections, 0.75)

plt.figure(figsize=(20,20))
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

masks_ls = [masks[i]['area'] for i in range(len(masks))]
plt.hist(masks_ls)
--------------------------------------------------------------------------

img_file = open("/home/zhi/data/PEAS/imgs.txt", "r")
img_content = img_file.read()
img_id = img_content.splitlines()
img_file.close()

with open("/home/zhi/data/PEAS/processed_data/img_id", "wb") as fp:   
    pickle.dump(img_id, fp)

with open("/home/zhi/data/PEAS/processed_data/img_id", "rb") as fp:
    img_id = pickle.load(fp)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/checkpoints/sam_vit_h_4b8939.pth"

f = []
id = []

num = len(img_id)

for i in range(num):
    f.append(img_id[i])
    id.append(img_id[i].split("/")[7][:9])
    print("\r Process{}%".format(round((i+1)*100/num)), end="")

f_id =  {'filename': f, 'id': id}
#df_f_id = pd.DataFrame(data = f_id)
df_f_id.to_csv('/home/zhi/data/PEAS/df_file_id.csv', index = False)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

num = len(img_id)
e = []
num = 5
for i in range(num):
    image = cv2.cvtColor(cv2.imread(img_id[i]), cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    e.append(np.squeeze(predictor.get_image_embedding().cpu().numpy()))
    print("\r Process{}%".format(round((i+1)*100/num)), end="")

#with open("/home/zhi/data/PEAS/processed_data/image_embedding_vit_h", "wb") as fp:   
 #   pickle.dump(e, fp)

with open("/home/zhi/data/PEAS/processed_data/image_embedding_vit_h", "rb") as fp:
    e = pickle.load(fp)

df_f_id = pd.read_csv('/home/zhi/data/PEAS/df_file_id.csv')

e_expand = [e[i].ravel() for i in range(len(e))]
e_array = np.array(e)
mapper = umap.UMAP().fit(np.array(e_expand))
umap.plot.points(mapper)

embedding = mapper.transform(np.array(e_expand))

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    embedding[:, 0], embedding[:, 1], cmap="Spectral", s=0.1
)
plt.title("UMAP", fontsize=18)

plt.show()

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(np.array(e_expand))

fig = px.scatter(
    projections, x=0, y=1,
    color=df_f_id.id
    #labels={'color': 'species'}
)
fig.show()

