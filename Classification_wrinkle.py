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
from sklearn.metrics import RocCurveDisplay

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

def CV(p_grid, out_fold, in_fold, model, X, y, rand):
    outer_cv = StratifiedKFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = StratifiedKFold(n_splits = in_fold, shuffle = True, random_state = rand)
    f1_train = []
    f1_test = []
    accuracy = []
    f1 = []
    recall = []
    precision_plot = []
    recall_plot = []
    precision = []
    spcificty = []
    AUPR = []
    AUROC = []
    y_true = []
    y_proba = []

    tprs = []
    fprs = []

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
        f = metrics.f1_score(y_test, y_pred)
        r = metrics.recall_score(y_test, y_pred, average='binary')
        p = metrics.precision_score(y_test, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        s = tn / (tn+fp)
        #print(a)
        #print(f)
        accuracy.append(a)
        f1.append(f)
        recall.append(r)
        precision.append(p)
        spcificty.append(s)
        y_score = clf.predict_proba(x_test)[:,1]
        y_true.append(y_test) 
        y_proba.append(y_score)
        #pre, re, thresholds = precision_recall_curve(y_test, y_score, pos_label = 1)
        #AUPR.append(auc(re, pre))
        #fpr, tpr, thresholds = roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)
        #fprs.append(fpr)
        #tprs.append(tpr)
        #AUROC.append(auc(fpr, tpr))
        
    return f1_train, f1_test, accuracy, f1, recall, precision, spcificty, y_true, y_proba

p_grid_rf = {'max_depth': [10, 50, 100, 200, None]}
model_rf = RandomForestClassifier()
p_grid_lsvm = {'base_estimator__C': [0.01, 0.5, 0.1, 1]}
model_lsvm = CalibratedClassifierCV(base_estimator=LinearSVC(max_iter = 10000))
p_grid_svm = {'C': [0.1, 1, 10], 'gamma': [1e-4, 1e-3, 'scale']}
model_svm =  SVC(kernel = 'rbf',  probability = True)

X = np.asarray(d_e_wrinkle['embeddings'])
y = np.array(d_e_wrinkle['wrinkling']).ravel()

RF_train_wrinkle, RF_test_wrinkle, acc_RF_wrinkle, f1_RF_wrinkle, recall_RF_wrinkle, precision_RF_wrinkle, spcificty_RF_wrinkle, ytrue_RF_wrinkle, y_proba_RF_wrinkle = CV(p_grid = p_grid_rf, out_fold = 5, in_fold = 5, model = model_rf, X = X, y =y, rand = 9)
RF_results = [RF_train_wrinkle, RF_test_wrinkle, acc_RF_wrinkle, f1_RF_wrinkle, recall_RF_wrinkle, precision_RF_wrinkle, spcificty_RF_wrinkle, ytrue_RF_wrinkle, y_proba_RF_wrinkle]
with open('/home/zhi/data/PEAS/PEAS_SAM/RF.txt', 'w') as file:
    for item_list in RF_results:
        row = ' '.join(map(str, item_list))  # Convert list to a space-separated string
        file.write(row + '\n')

LinSVM_train_wrinkle, LinSVM_test_wrinkle, acc_LinSVM_wrinkle, f1_LinSVM_wrinkle, recall_LinSVM_wrinkle, precision_LinSVM_wrinkle, spcificty_LinSVM_wrinkle, ytrue_LinSVM_wrinkle, y_proba_LinSVM_wrinkle = CV(p_grid = p_grid_lsvm, out_fold = 5, in_fold = 5, model = model_lsvm, X = X, y =y, rand = 9)
LinSVM_results = [LinSVM_train_wrinkle, LinSVM_test_wrinkle, acc_LinSVM_wrinkle, f1_LinSVM_wrinkle, recall_LinSVM_wrinkle, precision_LinSVM_wrinkle, spcificty_LinSVM_wrinkle, ytrue_LinSVM_wrinkle, y_proba_LinSVM_wrinkle]
with open('/home/zhi/data/PEAS/PEAS_SAM/LinSVM.txt', 'w') as file:
    for item_list in LinSVM_results:
        row = ' '.join(map(str, item_list))  # Convert list to a space-separated string
        file.write(row + '\n')

SVM_train_wrinkle, SVM_test_wrinkle, acc_SVM_wrinkle, f1_SVM_wrinkle, recall_SVM_wrinkle, precision_SVM_wrinkle, spcificty_SVM_wrinkle, ytrue_SVM_wrinkle, y_proba_SVM_wrinkle = CV(p_grid = p_grid_svm, out_fold = 5, in_fold = 5, model = model_svm, X = X, y =y, rand = 9)
SVM_results = [SVM_train_wrinkle, SVM_test_wrinkle, acc_SVM_wrinkle, f1_SVM_wrinkle, recall_SVM_wrinkle, precision_SVM_wrinkle, spcificty_SVM_wrinkle, ytrue_SVM_wrinkle, y_proba_SVM_wrinkle]
with open('/home/zhi/data/PEAS/PEAS_SAM/SVM.txt', 'w') as file:
    for item_list in SVM_results:
        row = ' '.join(map(str, item_list))  # Convert list to a space-separated string
        file.write(row + '\n')

fig, ax = plt.subplots()
ax.plot([0, 1], [1, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
y_real_rf = np.concatenate(ytrue_RF_wrinkle)
y_proba_rf = np.concatenate(y_proba_RF_wrinkle)
precision_rf, recall_rf, _ = precision_recall_curve(y_real_rf, y_proba_rf)
y_real_svm = np.concatenate(ytrue_SVM_wrinkle)
y_proba_svm = np.concatenate(y_proba_SVM_wrinkle)
precision_svm, recall_svm, _ = precision_recall_curve(y_real_svm, y_proba_svm)
y_real_lsvm = np.concatenate(ytrue_LinSVM_wrinkle)
y_proba_lsvm = np.concatenate(y_proba_LinSVM_wrinkle)
precision_lsvm, recall_lsvm, _ = precision_recall_curve(y_real_lsvm, y_proba_lsvm)
lab1 = 'RF AUC=%.4f' % (auc(recall_rf, precision_rf))
lab2 = 'SVM with Gaussian AUC=%.4f' % (auc(recall_svm, precision_svm))
lab3 = 'Linear SVM AUC=%.4f' % (auc(recall_lsvm, precision_lsvm))
ax.step(recall_rf, precision_rf, label=lab1, lw=2, color='blue')
ax.step(recall_svm, precision_svm, label=lab2, lw=2, color='orange')
ax.step(recall_lsvm, precision_lsvm, label=lab3, lw=2, color='brown')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall (PR) Curve')
ax.legend(loc='lower left', fontsize='small')


fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
y_real_rf = np.concatenate(ytrue_RF_wrinkle)
y_proba_rf = np.concatenate(y_proba_RF_wrinkle)
fpr_rf, tpr_rf, _ = roc_curve(y_real_rf, y_proba_rf)
y_real_svm = np.concatenate(ytrue_SVM_wrinkle)
y_proba_svm = np.concatenate(y_proba_SVM_wrinkle)
fpr_svm, tpr_svm, _ = roc_curve(y_real_svm, y_proba_svm)
y_real_lsvm = np.concatenate(ytrue_LinSVM_wrinkle)
y_proba_lsvm = np.concatenate(y_proba_LinSVM_wrinkle)
fpr_lsvm, tpr_lsvm, _ = roc_curve(y_real_lsvm, y_proba_lsvm)
lab1 = 'RF AUC=%.4f' % (auc(fpr_rf, tpr_rf))
lab2 = 'SVM with Gaussian AUC=%.4f' % (auc(fpr_svm, tpr_svm))
lab3 = 'Linear SVM AUC=%.4f' % (auc(fpr_lsvm, tpr_lsvm))
ax.step(fpr_rf, tpr_rf, label=lab1, lw=2, color='blue')
ax.step(fpr_svm, tpr_svm, label=lab2, lw=2, color='orange')
ax.step(fpr_lsvm, tpr_lsvm, label=lab3, lw=2, color='brown')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right', fontsize='small')