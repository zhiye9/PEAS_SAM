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
import umap
import umap.plot
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
df_wrinkle = pd.read_excel('/home/zhi/nas/PEAS/Nordgen_wrinkledpeas2023.xlsx', usecols = 'A, B')

e_expand = [e[i].ravel() for i in range(len(e))]

idd = []
e_idx = []
wrinkle = []

for j in range(df_wrinkle.shape[0]):
    index = df_f_id.index[df_f_id['id'] == df_wrinkle['Accession number'][j]]
    if len(index) != 0:
        idd.append(df_f_id['id'].loc[index[0]])
        e_idx.append(e_expand[index[0]])
        wrinkle.append(df_wrinkle['Seed wrinkling'].loc[j])
        print("\r Process{}%".format(round((j+1)*100/df_wrinkle.shape[0])), end="")

d_e_wrinkle = {'id':idd, 'wrinkling': wrinkle, 'embeddings':np.vstack(e_idx)}
#df_e_wrinkle = pd.DataFrame(data = d_e_wrinkle)
df_f_id.index[df_f_id['id'] == df_wrinkle['Accession number'][1]]

e_array = np.array(e)
mapper = umap.UMAP().fit(np.array(d_e_wrinkle['embeddings']))
umap.plot.points(mapper)

embedding = umap.UMAP().fit_transform(np.array(d_e_wrinkle['embeddings']))
umap.plot.points(embedding, labels = d_e_wrinkle['wrinkling'])

embedding_wrinkle = np.column_stack([embedding, np.array(wrinkle)])

fig, ax = plt.subplots(figsize=(6, 5))
scatter = plt.scatter(
    embedding_wrinkle[:, 0], embedding_wrinkle[:, 1], cmap="Spectral", c = embedding_wrinkle[:, 2], 
)
classes = ['Wrinkle absent', 'Wrinkle present']
plt.title("UMAP", fontsize=12)
plt.legend(handles=scatter.legend_elements()[0], loc ='upper right', labels = classes, fontsize = 9)
plt.show()

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(np.array(d_e_wrinkle['embeddings']))
projections_wrinkle = np.column_stack([projections, np.array(wrinkle)])

fig, ax = plt.subplots(figsize=(6, 5))
scatter = plt.scatter(
    projections_wrinkle[:, 0], projections_wrinkle[:, 1], cmap="Spectral", c = projections_wrinkle[:, 2], 
)
classes = ['Wrinkle absent', 'Wrinkle present']
plt.title("t-SNE", fontsize=12)
plt.legend(handles=scatter.legend_elements()[0], loc ='upper right', labels = classes, fontsize = 9)
plt.show()


fig = px.scatter(
    projections, x=0, y=1,
    color=wrinkle
    #labels={'color': 'species'}
)
fig.show()

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    embedding[:, 0], embedding[:, 1], cmap="Spectral", s=0.1
)
plt.title("TSNE", fontsize=18)
