import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import nibabel as nib
from tqdm import tqdm
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from numpy.random import Generator, PCG64DXSM
import time
import random

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

n_components: int = 3
max_iter: int = 100
change_tolerance: float = 1e-6
seed = 123

def maximation(x,n_samples,posteriors):
    
    labels = np.zeros((x.shape[0], n_components))
    labels[np.arange(n_samples), np.argmax(posteriors, axis=1)] = 1

        
    posteriors = posteriors * labels
    counts = np.sum(posteriors, 0)
    weithed_avg = np.dot(posteriors.T, x)

    means = weithed_avg / counts[:, np.newaxis]

    sigmas = np.zeros((n_components, x.shape[1], x.shape[1]))
    for i in range(n_components):
        diff = x - means[i, :]
        weighted_diff = posteriors[:, i][:, np.newaxis] * diff
        sigmas[i] = np.dot(weighted_diff.T, diff) / counts[i]

    priors = counts / len(x)
        
    return means,sigmas, priors
    
        
def expectation(x,means,sigmas,priors):
    
    n_components, _ = means.shape
    likelihood = [
        multivariate_normal.pdf(x, means[i, :], sigmas[i, :, :], allow_singular=True) for i in range(n_components)]
    likelihood = np.asarray(likelihood).T
    num = np.asarray([ 
                      likelihood[:, j] * priors[j] for j in range(n_components)]).T
    denom = np.sum(num, 1)
    
    posteriors = np.asarray([num[:, j] / denom for j in range(n_components)]).T
    
    return posteriors, likelihood

def create_masks():
    for i in tqdm(range(5)):
        brain_mask_img = nib.load(os.path.join('..','P2_Data',f'{i}','LabelsForTesting.nii'))
        brain_mask_array = brain_mask_img.get_fdata()
        bm_array = np.where(brain_mask_array > 0, 255, 0).astype('uint8')
        clipped_img = nib.Nifti1Image(bm_array, brain_mask_img.affine)
        nib.save(clipped_img,os.path.join('..','P2_Data',f'{i}','brainMask.nii'))

def min_max_normalization(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img

def get_initial_values(x,labels,cov_reg=1e-6):
    n_components = labels.shape[1]
    n_feat = x.shape[1]
    min_val = 10 * np.finfo(labels.dtype).eps
    counts = np.sum(labels, axis=0) + min_val
    means = np.dot(labels.T, x) / counts[:, np.newaxis]
    sigmas = np.zeros((n_components, n_feat, n_feat))
    for i in range(n_components):
        diff = x - means[i, :]
        sigmas[i] = np.dot(
        (labels[:, i][:, np.newaxis] * diff).T, diff) / counts[i]
        sigmas[i].flat[:: n_feat + 1] += cov_reg
        if np.ndim(sigmas) == 1:
            sigmas = (sigmas[:, np.newaxis])[:, np.newaxis]
    return means, sigmas, counts

def get_tissue(t1,t2,brain_mask,type='knn',operation='EM'):

    
    t1_array = min_max_normalization(t1)
    t2_array = min_max_normalization(t2)
    
    t1_vector = t1_array[brain_mask == 255].flatten()
    t2_vector = t2_array[brain_mask == 255].flatten()
    
    data = np.array(t1_vector)[:, np.newaxis]
    
    if operation == 'EM':
    
        n_feat = data.shape[1] if np.ndim(data) > 1 else 1
        n_samples = len(data)
        labels = np.zeros((n_samples, n_components))
    
        if type == 'knn':
            kmeans = KMeans(
            n_clusters=n_components, 
            random_state=seed).fit(data)
            labels[np.arange(n_samples), kmeans.labels_] = 1
        elif type == 'random':
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(n_components, size=n_samples)
            labels[np.arange(n_samples),idx] = 1
        
        means, sigmas, counts = get_initial_values(data,labels)
    
        priors = np.ones((n_components, 1)) / n_components
    
        prev_log_lkh = 0
    
        start = time.time()
    
        for it in tqdm(range(max_iter)):
            n_iter_ = it + 1

            posteriors, likelihood  = expectation(data,means,sigmas,priors)

            for i in range(n_components):
                likelihood[:, i] = likelihood[:, i] * priors[i]
        
            log_lkh = np.sum(np.log(np.sum(likelihood, 1)), 0)
            difference = abs(prev_log_lkh - log_lkh)
            prev_log_lkh = log_lkh
        
            if difference < change_tolerance:
                break
            
            means, sigmas, priors = maximation(data,n_samples,posteriors)
        
        cluster_centers = means
    
        posteriors, likelihood  = expectation(data,means,sigmas,priors)
    
        preds = np.argmax(posteriors, 1)
    
        predictions = brain_mask.flatten()
        predictions[predictions == 255] = preds + 1
    
        t1_seg_res = predictions.reshape(t1.shape)
    
        t1_t = time.time() - start
        
    if operation == 'Kmeans':
        start = time.time()
        model = KMeans(
        n_clusters=n_components, 
        random_state=seed).fit(data)
        preds = model.predict(data)
        predictions = brain_mask.flatten()
        predictions[predictions == 255] = preds + 1
        t1_seg_res = predictions.reshape(t1.shape)
        t1_t = time.time() - start
    
    data = np.array([t1_vector, t2_vector]).T  
    
    if operation == 'EM':
    
        n_feat = data.shape[1] if np.ndim(data) > 1 else 1
        n_samples = len(data)
        labels = np.zeros((n_samples, n_components))
    
        if type == 'knn':
            kmeans = KMeans(
            n_clusters=n_components, 
            random_state=seed).fit(data)
            labels[np.arange(n_samples), kmeans.labels_] = 1
        elif type == 'random':
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(n_components, size=n_samples)
            labels[np.arange(n_samples),idx] = 1   
         
        means, sigmas, counts = get_initial_values(data,labels)
        
        priors = np.ones((n_components, 1)) / n_components
        
        prev_log_lkh = 0
        
        start = time.time()
        
        for it in tqdm(range(max_iter)):
            n_iter_ = it + 1

            posteriors, likelihood  = expectation(data,means,sigmas,priors)

            for i in range(n_components):
                likelihood[:, i] = likelihood[:, i] * priors[i]
                
            log_lkh = np.sum(np.log(np.sum(likelihood, 1)), 0)
            difference = abs(prev_log_lkh - log_lkh)
            prev_log_lkh = log_lkh
            if difference < change_tolerance:
                break
            
            means, sigmas, priors = maximation(data,n_samples,posteriors)
            
        cluster_centers = means
        
        posteriors, likelihood  = expectation(data,means,sigmas,priors)
        
        preds = np.argmax(posteriors, 1)
        
        predictions = brain_mask.flatten()
        predictions[predictions == 255] = preds + 1
        
        t2_seg_res = predictions.reshape(t1.shape)  
        
        t2_t = time.time() - start
    
    if operation == 'Kmeans':
        start = time.time()
        model = KMeans(
        n_clusters=n_components, 
        random_state=seed).fit(data)
        preds = model.predict(data)
        predictions = brain_mask.flatten()
        predictions[predictions == 255] = preds + 1
        t2_seg_res = predictions.reshape(t1.shape)
        t2_t = time.time() - start
    
    return t1_seg_res, t2_seg_res, t1_t, t2_t

def match_pred_with_gt(pred,gt):
    wm = np.zeros_like(pred)
    gm = np.zeros_like(pred)
    cfs = np.zeros_like(pred)
    for tets_prob in range(3):
        probs = []
        for prob in range(3):
            gt_layer = np.where(gt==prob+1, 1, 0)
            test = np.where(pred==tets_prob+1,1,0)
            intersection = np.sum(np.logical_and(gt_layer, test))
            union = np.sum(np.logical_or(gt_layer, test))
            iou = intersection / union
            probs.append(iou)
        
        max_location = probs.index(max(probs))
        
        if (max_location+1) == 1:
            cfs[pred==tets_prob+1] = max_location +1 
        if (max_location+1) == 2:
            gm[pred==tets_prob+1] = max_location+1
        if (max_location+1) == 3:
            wm[pred==tets_prob+1] = max_location+1

    return wm, gm, cfs


def plots(volumes, names, slice_n: int = 25):
    n = len(volumes)
    fig, ax = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        ax[i].set_title(names[i])
        ax[i].imshow(volumes[i][slice_n, :, :], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
    
def dice_score(gt, pred):
    classes = np.unique(gt[gt != 0])
    dice = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        dice[i-1] = np.sum(bin_pred[bin_gt == 1]) * 2.0 / (np.sum(bin_pred) + np.sum(bin_gt))
    return dice.tolist()
    
    

        