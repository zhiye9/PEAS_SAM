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
from sklearn.manifold import TSNE
import plotly.express as px
from scipy import ndimage
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle


img_temp =  '/home/zhi/nas/PEAS/PEAS_images/OneDrive_6_23-03-2023/NGB106117_1_8_SSD_seed_1.png'
img_temp =  '/home/zhi/nas/PEAS/PEAS_images/OneDrive_4_23-03-2023/NGB101283_1_2_SD_seed_1.png'
img_temp =  '/home/zhi/nas/PEAS/PEAS_images/OneDrive_4_23-03-2023/NGB101288_1_2_SD_seed_1.png'
img_temp =  '/home/zhi/nas/PEAS/PEAS_images/OneDrive_4_23-03-2023/NGB101475_1_2_SD_seed_1.png'

imagebgr = cv2.imread(img_temp)
image = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/zhi/data/PEAS/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh=0.2, min_mask_region_area=600)

m = mask_sorted[6]['segmentation'].astype(np.uint8)
img2 = imagebgr.copy()
black_background = np.zeros(shape=m.shape, dtype=np.uint8)
black_background = np.full(shape=m.shape, fill_value = 255, dtype=np.uint8)
res = cv2.bitwise_not(img2, black_background, mask = m)

masked_image = cv2.bitwise_and(image, img2, mask=m)

plt.figure(figsize=(20,20))
plt.imshow(res)
plt.axis('off')
plt.show()

masks_area = [[i['area'] for i in sorted(masks, key=lambda x: x['area'], reverse=False)]]
with open("/home/zhi/data/PEAS/processed_data/masks_area", "wb") as fp:   
    pickle.dump(masks_area, fp)


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
print(masks[0].keys())

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(sam_result=masks)
annotated_image = mask_annotator.annotate(scene=imagebgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[imagebgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

maskes = [
    mask['segmentation']
    for mask
    in sorted(masks, key=lambda x: x['area'], reverse=True)
]

sv.plot_images_grid(
    images=maskes,
    grid_size=(6, int(len(masks) / 6)+1),
    size=(16, 16)
)



mask_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
binary_mask = mask_sorted[1]['segmentation']

new_mask = []
for single_mask in mask_sorted:
    if (single_mask['area'] < 25000) & (single_mask['area'] > 2000):
        regions = regionprops(label(single_mask['segmentation']))
        if len(regions) == 1:      
            img2 = imagebgr.copy()
            m = single_mask['segmentation'].astype(np.uint8)
            black_background = np.zeros(shape=m.shape, dtype=np.uint8)
            res = cv2.bitwise_not(img2, black_background, mask = m)
            indict = 0
            for j in range(3):
                num_unique = np.unique(res[:, :, j], return_counts= True)
                if (255 in num_unique[0]) & (len(num_unique[0]) == 2):
                    indict += 1
                elif (255 in num_unique[0]) & (num_unique[1][-1] == sorted(num_unique[1], reverse = True)[1]) & (num_unique[1][-1] > sorted(num_unique[1], reverse = True)[2]*50):
                    indict += 1
            if indict != 3:

                new_mask.append(single_mask)

    return regions[0].eccentricity
mask_eccentricity = regions[0].eccentricity 
-------------------------------------------------------------------------------------------
# Load the image using Pillow
fill_color = (255, 255, 255)

imagebgr = cv2.imread(img_temp)

# Define the coordinates for the rectangular bounding boxes (x, y, width, height)
bounding_boxes = [
    (0, 0, 200, 100),  # Example: Top-left corner
    (0, imagebgr.shape[0] - 95, 295, 95),  # Example: Bottom-left corner
    (imagebgr.shape[1] - 320, imagebgr.shape[0]-100, 320, 100),  # Example: Bottom-right corner
]

for box in bounding_boxes:
    x, y, width, height = box
    imagebgr[y:y + height, x:x + width] = fill_color

image = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()
---------------------------------------------------------------------------------------------
mask_idx = [mask_sorted[n] for n in idx]

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(masks)
#detections = sv.Detections.from_sam(mask_idx)
annotated_image = mask_annotator.annotate(image, detections, 0.75)

plt.figure(figsize=(20,20))
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

masks_ls = [masks[i]['area'] for i in range(len(masks))]
plt.hist(masks_ls)

maskes = [
    mask_idx['segmentation']
    for mask_idx
    in sorted(mask_idx, key=lambda x: x['area'], reverse=True)
]

sv.plot_images_grid(
    images=maskes,
    grid_size=(6, int(len(mask_idx) / 6)+1),
    size=(16, 16)
)

--------------------------------------------------------------------------
single_mask_segmentation = mask_idx[157]['segmentation'].astype(np.uint8)
contours, _ = cv2.findContours(single_mask_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_object_masks = []
for contour in contours:
    # Approximate the contour with a simpler polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    valid_object_masks.append(contour)
    # Check if the approximation has straight lines (e.g., has more than 4 vertices)
    if len(approx) > 4:
        print('!!!!!!!!!!!!!!')
        valid_object_masks.append(contour)
plt.imshow(single_mask_segmentation, cmap='gray')

# Plot the detected contours on top of the image
for contour in valid_object_masks:
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='red', linewidth=2)

plt.show()

for contour in contours:
    contour = contour.squeeze()  # Remove the extra dimension
    plt.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)

plt.show()

height, width = image.shape[:2]

# Create a list to store valid object masks
valid_object_masks = []

# Loop through the detected contours
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Check if the bounding box is entirely within the image
    if (x > 0 and y > 0 and x + w < width and y + h < height):
        print("!")
        valid_object_masks.append(contour)

# Create an empty mask to store the valid object masks
valid_objects_mask = np.zeros_like(binary_mask)

# Draw the valid object contours on the mask
cv2.drawContours(valid_objects_mask, valid_object_masks, -1, (255), thickness=cv2.FILLED)

plt.imshow(single_mask_segmentation, cmap='gray')
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    rectangle = Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rectangle)

plt.show()

def detect_boundary(single_mask_segmentation):
    contours, _ = cv2.findContours(single_mask_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the bounding box is entirely within the image
        if (x > 0 and y > 0 and x + w < width and y + h < height):

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
id_d = []

num = len(img_id)

for i in range(num):
    f.append(img_id[i])
    id.append(img_id[i].split("/")[7][:9])
    id_d.append(img_id[i].split("/")[6] + '/' + img_id[i].split("/")[7])
    print("\r Process{}%".format(round((i+1)*100/num)), end="")

f_id =  {'filename': f, 'id_in_dictionary': id_d, 'id': id}
df_f_id = pd.DataFrame(data = f_id)
df_f_id.to_csv('/home/zhi/data/PEAS/df_file_id.csv', index = False)

-----------------------------------------------------------------------------------------------------
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

f1 = []
id1 = []
id2 = []
id_d = []
num = len(img_id)

for i in range(num):
    f1.append(img_id[i])
    id1.append(extract_substring(img_id[i]))
    id2.append(extract_substring(img_id[i], NGB = False))
    id_d.append(img_id[i].split("/")[6] + '/' + img_id[i].split("/")[7])
    print("\r Process{}%".format(round((i+1)*100/num)), end="")

f_id1 = {'filename': f1, 'id_in_dictionary': id_d, 'id': id1, 'acc_num':id2}
df_f_id1 = pd.DataFrame(data = f_id1)
df_f_id1.to_csv('/home/zhi/data/PEAS/df_file_id1.csv', index = False)
-----------------------------------------------------------------------------------------------------

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

df_f_id = pd.read_csv('/home/zhi/data/PEAS/df_file_id1.csv')
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
plt.legend(handles=scatter.legend_elements()[0], loc ='lower left', labels = classes, fontsize = 9)
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

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def CV(p_grid, out_fold, in_fold, model, X, y, rand):
    outer_cv = StratifiedKFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = StratifiedKFold(n_splits = in_fold, shuffle = True, random_state = rand)
    f1train = []
    f1test = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "f1")
        clf.fit(x_train, y_train)
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        #r2train.append(mean_squared_error(y_train, y_pred))
        f1_train.append(f1_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #r2test.append(mean_squared_error(y_test, y_pred))
        f1_test.append(f1_score(y_test, y_pred))
        a = metrics.accuracy_score(y_test, y_pred)
        f = metrics.f1_score(y_test, y_pred, pos_label = '1')
        r = metrics.recall_score(y_test, y_pred, average='binary', pos_label = '1')
        p = metrics.precision_score(y_test, y_pred, average='binary', pos_label = '1')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        s = tn / (tn+fp)
        #print(a)
        #print(f)
        accuracy.append(a)
        f1.append(f)
        recall.append(r)
        precision.append(p)
        spcificty.append(s)
        y_score = classifier.predict_proba(x_test)[:,1]
        y_true.append(y_test) 
        y_proba.append(y_score)
        pre, re, thresholds = precision_recall_curve(y_test, y_score, pos_label = '1')
        AUPR.append(auc(re, pre))
        fpr, tpr, thresholds = roc_curve(y_test, y_score, drop_intermediate = False, pos_label = '1')
        AUROC.append(auc(fpr, tpr))
        
    return f1_train, f1_test, accuracy, f1, recall, precision, spcificty, AUPR, AUROC

p_grid_rf = {'randomforestclassifier__max_depth': [10, 50, 100, 200, None]}
model_rf = RandomForestClassifier()
p_grid_lsvm = {'base_estimator__C': [0.01, 0.5, 0.1, 1, 10]}
model_lsvm = CalibratedClassifierCV(base_estimator=LinearSVC(max_iter = 10000))
p_grid_svm = {'svc__C': [0.1, 1, 10], 'svc__gamma': [1e-3, 1e-4, 'scale']}
model_svm =  SVC(kernel = 'rbf',  probability = True)

X = np.asarray(d_e_wrinkle['embeddings'])
y = np.array(d_e_wrinkle['wrinkling']).ravel()

RF_train_wrinkle, RF_test_wrinkle = CV(p_grid = p_grid_rf, out_fold = 5, in_fold = 5, model = model_rf, X = X, y =y, rand = 9)
LinSVM_train_wrinkle, LinSVM_test_wrinkle = CV(p_grid = p_grid_lsvm, out_fold = 5, in_fold = 5, model = model_lsvm, X = X, y =y, rand = 9)
SVM_train_wrinkle, SVM_test_wrinkle = CV(p_grid = p_grid_svm, out_fold = 5, in_fold = 5, model = model_svm, X = X, y =y, rand = 9)

accuracy = []
f1 = []
recall = []
precision_plot = []
recall_plot = []
precision = []
spcificty = []
tprs = []
fprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
AUROCl = []
AUPRl = []
y_truel = []
y_probal = []
AUROC = []
AUPR = []
y_true = []
y_proba = []
AUROCr = []
AUPRr = []
y_truer = []
y_probar = []


outer_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 0)
inner_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 0)
for j in range(5):
    train, test = list(outer_cv.split(filename, y))[j]
    x_train, x_test = ica[train], ica[test]
    y_train, y_test = y[train], y[test]
    kinds = ['correlation', 'partial correlation', 'tangent']
    connectivity = ConnectivityMeasure(kind=kinds[2], vectorize=True)
    X = connectivity.fit_transform(x_train)
    Y = connectivity.transform(x_test)
    # fit the classifier
    classifier = make_pipeline(StandardScaler(), model)
    clf = GridSearchCV(estimator = classifier, param_grid = p_grid, cv = inner_cv, scoring = "accuracy")
    clf.fit(X, y_train)
    print(clf.best_estimator_)
    
#model =  SVC(kernel = 'rbf', gamma = 1e-3, C = 10, probability = True)
model =  SVC(kernel = 'rbf',  probability = True)
model = CalibratedClassifierCV(base_estimator=LinearSVC(max_iter = 10000))
model = RandomForestClassifier(criterion = 'entropy')
outer_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 0)
inner_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 0)
for j in range(5):
    train, test = list(outer_cv.split(filename, y))[j]
    x_train, x_test = ica[train], ica[test]
    y_train, y_test = y[train], y[test]
    kinds = ['correlation', 'partial correlation', 'tangent']
    connectivity = ConnectivityMeasure(kind=kinds[2], vectorize=True)
    X = connectivity.fit_transform(x_train)
    Y = connectivity.transform(x_test)
    # fit the classifier
    classifier = make_pipeline(StandardScaler(), model)
    classifier.fit(X, y_train)
    # make predictions for the left-out test subjects
    y_pred = classifier.predict(Y)
    a = metrics.accuracy_score(y_test, y_pred)
    f = metrics.f1_score(y_test, y_pred, pos_label = '1')
    r = metrics.recall_score(y_test, y_pred, average='binary', pos_label = '1')
    p = metrics.precision_score(y_test, y_pred, average='binary', pos_label = '1')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    s = tn / (tn+fp)
    #print(a)
    #print(f)
    accuracy.append(a)
    f1.append(f)
    recall.append(r)
    precision.append(p)
    spcificty.append(s)
    y_score = classifier.predict_proba(Y)[:,1]
    y_true.append(y_test) 
    y_proba.append(y_score)
    pre, re, thresholds = precision_recall_curve(y_test, y_score, pos_label = '1')
    AUPR.append(auc(re, pre))
    fpr, tpr, thresholds = roc_curve(y_test, y_score, drop_intermediate = False, pos_label = '1')
    AUROC.append(auc(fpr, tpr))
print(np.mean(accuracy))
print(np.std(accuracy))

print(np.mean(f1))
print(np.std(f1))

print(np.mean(recall))
print(np.std(recall))

print(np.mean(precision))
print(np.std(precision))

print(np.mean(spcificty))
print(np.std(spcificty))

print(np.mean(AUROCr))
print(np.std(AUROC))

print(np.mean(AUPRr))
print(np.std(AUPR))

def read_aslist(file_path, results):
    # Open the file and read its contents line by line
    with open(file_path, "r") as file:
        for line in file:
            # Split each line into a list of elements separated by spaces
            elements = [float(element) for element in line.strip().split()]
            results.append(elements)
    return results

file_path = "/home/zhi/data/PEAS/PEAS_SAM/RF.txt"
RF_results = read_aslist(file_path, [])

file_path = "/home/zhi/data/PEAS/PEAS_SAM/LinSVM.txt"
LinSVM_results = read_aslist(file_path, [])

file_path = "/home/zhi/data/PEAS/PEAS_SAM/SVM.txt"
SVM_results = read_aslist(file_path, [])

labels = ['RandomForest', 'Linear SVM', 'SVM with Gaussian kernel']
dpi = 1000
title = 'Classification results of wrinkled peas'

x = np.arange(len(labels))
train_mean = [round(np.mean(RF_results[0]), 3), round(np.mean(LinSVM_results[0]), 3), round(np.mean(SVM_results[0]), 3)]
test_mean = [round(np.mean(RF_results[1]), 3), round(np.mean(LinSVM_results[1]), 3), round(np.mean(SVM_results[1]), 3)]
train_std = [np.std(RF_results[0]),  np.std(LinSVM_results[0]), np.std(SVM_results[0])]
test_std = [np.std(RF_results[1]), np.std(LinSVM_results[1]), np.std(SVM_results[1])]    

train_mean = [round(np.mean(RF_results[0]), 3), round(np.mean(LinSVM_results[0]), 3), round(np.mean(SVM_results[0]), 3)]
test_mean = [round(np.mean(RF_results[1]), 3), round(np.mean(LinSVM_results[1]), 3), round(np.mean(SVM_results[1]), 3)]
train_std = [np.std(RF_results[0]),  np.std(LinSVM_results[0]), np.std(SVM_results[0])]
test_std = [np.std(RF_results[1]), np.std(LinSVM_results[1]), np.std(SVM_results[1])]    

fig, ax = plt.subplots(dpi = 1000)

width = 0.35
rects1 = ax.bar(x - width/2, train_mean, width, yerr = train_std, label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, test_mean, width, yerr = test_std, label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
#plt.xticks(rotation=60)
#plt.yticks(np.linspace(0.0,0.7,0.1))
ax.set_xticklabels(labels, fontsize = 6)
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()   
