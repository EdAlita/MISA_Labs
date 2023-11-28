import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from tqdm import tqdm
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import time
from scipy.stats import mode

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class maximum_expectation_algorithm:
    def __init__(self, expected_components=3, maximum_iteration=300, change_tolerance=1e-6, seed=123):
        self.expected_components = expected_components
        self.maximum_iteration = maximum_iteration
        self.change_tolerance = change_tolerance
        self.seed = seed

    def maximation_phase(self, x, posteriors):

       # Get means
        counts = np.sum(posteriors, 0)
        weithed_avg = np.dot(posteriors.T, x)
        means = weithed_avg / counts[:, np.newaxis] +  1e-8 


        sigmas = np.zeros((self.expected_components, x.shape[1], x.shape[1]))
        for i in range(self.expected_components):
            difference = x - means[i, :]
            weighted_diff = posteriors[:, i][:, np.newaxis] * difference
            sigmas[i] = np.dot(weighted_diff.T, difference) / counts[i]
            priors = counts / len(x)
            
        return means, sigmas, priors

    def expectation_phase(self, x, means, sigmas, priors,atlas_use,atlas_map):
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("Input data contains NaN or Inf values.")
        if np.any(np.isnan(means)) or np.any(np.isinf(means)):
            raise ValueError("Means contains NaN or Inf values.")
        if np.any(np.isnan(sigmas)) or np.any(np.isinf(sigmas)):
            raise ValueError("Sigmas contains NaN or Inf values.")

        expected_components, _ = means.shape
        likelihood = [multivariate_normal.pdf(
            x, means[i, :], sigmas[i, :, :], allow_singular=True)for i in range(expected_components)]
        likelihood = np.asarray(likelihood).T
        num = np.asarray(
            [likelihood[:, j] * priors[j] for j in range(expected_components)]).T
        
        denom = np.sum(num, 1)

        posteriors = np.asarray([num[:, j] / denom for j in range(expected_components)]).T

        if( atlas_use == 'into'):
            posteriors = posteriors * atlas_map
        
        return posteriors, likelihood

    def get_initial_values(self, x, labels, covenge_regression=1e-9):
        expected_components = labels.shape[1]
        number_features = x.shape[1]
        minimum_value = 10 * np.finfo(labels.dtype).eps
        counts = np.sum(labels, axis=0) + minimum_value
        means = np.dot(labels.T, x) / counts[:, np.newaxis]
        sigmas = np.zeros((expected_components, number_features, number_features))
        for i in range(expected_components):
            difference = x - means[i, :]
            sigmas[i] = np.dot((labels[:, i][:, np.newaxis] * difference).T, difference) / counts[i]
            sigmas[i].flat[::number_features + 1] += covenge_regression
            if np.ndim(sigmas) == 1:
                sigmas = (sigmas[:, np.newaxis])[:, np.newaxis]
        return means, sigmas

    def tissue_segmentation(self, T1, mask, type, operation='EM', label_pro= None, tissue_model = None, atlas_use=None):
        t1_vector = T1[mask == 255].flatten()
        data_vector = np.array(t1_vector)[:, np.newaxis]
        if label_pro is not None:
            atlas_map_vector = label_pro[:, mask == 255].reshape(-1,label_pro.shape[0])
            #atlas_map_vector = atlas_map_vector[1:, :]
        else:
            atlas_map_vector = None
        
        

        if operation == 'EM':
            number_samples = len(data_vector)
            posteriors = np.zeros((number_samples, self.expected_components))

            if type == 'knn':
                kmeans = KMeans(n_clusters=self.expected_components, random_state=self.seed).fit(data_vector)
                posteriors[np.arange(number_samples), kmeans.labels_] = 1            
            elif type == 'label_propragation':
                posteriors = atlas_map_vector
            elif type == 'random':
                rng = np.random.default_rng(seed=self.seed)
                idx = rng.choice(self.expected_components, size=number_samples, replace=False)
                posteriors[idx, np.arange(number_samples)] = 1
                priors = np.ones((number_samples, 1)) / self.expected_components
            else:
                tissue_prob_maps = np.zeros((number_samples, self.expected_components))
                for i in range(number_samples):
                    tissue_prob_maps[i, 0] = tissue_model[0, :][data_vector[i,0]]
                    tissue_prob_maps[i, 1] = tissue_model[1, :][data_vector[i,0]]
                    tissue_prob_maps[i, 2] = tissue_model[2, :][data_vector[i,0]]
                posteriors = tissue_prob_maps





            means, sigmas, priors = self.maximation_phase(data_vector, posteriors)

            previous_log_likelihood = 0

            start_time = time.time()

            for it in tqdm(range(self.maximum_iteration),desc=f'EM in progress with {type} init and {atlas_use} atlas'):
                posteriors, likelihood = self.expectation_phase(data_vector, means, sigmas, priors,atlas_use,atlas_map_vector)

                for i in range(self.expected_components):
                    likelihood[:, i] = likelihood[:, i] * priors[i]

                log_likelihood = np.sum(np.log(np.sum(likelihood, 1)), 0)
                difference = abs(previous_log_likelihood - log_likelihood)
                previous_log_likelihood = log_likelihood

                if difference < self.change_tolerance:
                    break

                means, sigmas, priors = self.maximation_phase(data_vector, posteriors)

            posteriors, likelihood = self.expectation_phase(data_vector, means, sigmas, priors,atlas_use,atlas_map_vector)

            if(atlas_use == 'after'):
                posteriors = posteriors * atlas_map_vector

            predictions = np.argmax(posteriors, 1)
            predictions_image = mask.flatten()
            predictions_image[predictions_image == 255] = predictions + 1

            t1_segmentation_result = predictions_image.reshape(T1.shape)

            t1_time = time.time() - start_time

        return t1_segmentation_result, t1_time


    def min_max_norm(self, img, max_val, mask, dtype):
        if mask is not None:
            img_ = img.copy()
            img = img[mask != 0].flatten()
        if max_val is None:
            max_val = np.iinfo(img.dtype).max
        img = (img - img.min()) / (img.max() - img.min()) * max_val
        if mask is not None:
            img_[mask != 0] = img.copy()
            img = img_.copy()
        if dtype is not None:
            return img.astype(dtype)
        else:
            return img

    def match_pred_with_gt(self, seg, gt, prob_array= None):
        shape = seg.shape
        seg = seg.flatten()
        gt = gt.flatten()
        order = {}
        for label in [0, 2, 3]:
            labels, counts = np.unique(seg[gt == label], return_counts=True)
            order[label] = labels[np.argmax(counts)]
        order[1] = [i for i in [0, 1, 2, 3] if i not in list(order.values())][0]
        seg_ = seg.copy()
        prob_array_ = prob_array.copy() if prob_array is not None else None
        for des_val, seg_val in order.items():
            seg[seg_ == seg_val] = des_val
            if des_val in [1, 2, 3]:
                if prob_array_ is not None:
                    prob_array[:, des_val-1] = prob_array_[:, seg_val-1]
        if prob_array is not None:
            return seg.reshape(shape), prob_array
        else:
            return seg.reshape(shape)

    def dice_score(self, ground_truth, prediction):
        dice = np.zeros((3))
        for i in [1, 2, 3]:
            bin_pred = np.where(prediction == i, 1, 0)
            bin_gt = np.where(ground_truth == i, 1, 0)
            dice[i-1] = np.sum(bin_pred[bin_gt == 1]) * 2.0 / (np.sum(bin_pred) + np.sum(bin_gt))
        return dice.tolist()

    def create_masks(self):
        for i in tqdm(range(5),desc='Creating masks....'):
            brain_mask_img = nib.load(os.path.join('..', 'P2_Data', f'{i}', 'LabelsForTesting.nii'))
            brain_mask_array = brain_mask_img.get_fdata()
            brain_mask_array = np.where(brain_mask_array > 0, 255, 0).astype('uint8')
            mask_image = nib.Nifti1Image(brain_mask_array, brain_mask_img.affine)
            nib.save(mask_image, os.path.join('..', 'P2_Data', f'{i}', 'brainMask.nii'))

    def get_plots(self, volumes, names, slice_to_display: int = 25):
        n = len(volumes)
        fig, ax = plt.subplots(1, n, figsize=(20, 5))
        for i in range(n):
            ax[i].set_title(names[i])
            ax[i].imshow(volumes[i][slice_to_display, :, :], cmap='gray')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.show()