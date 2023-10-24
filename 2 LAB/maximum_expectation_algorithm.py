import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import nibabel as nib
from tqdm import tqdm
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import time

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class maximum_expectation_algorithm:
    def __init__(self, expected_components=3, maximum_iteration=100, change_tolerance=1e-6, seed=123):
        self.expected_components = expected_components
        self.maximum_iteration = maximum_iteration
        self.change_tolerance = change_tolerance
        self.seed = seed

    def maximation_phase(self, x, posteriors):
        labels = np.zeros((x.shape[0], self.expected_components))
        labels[np.arange(x.shape[0]), np.argmax(posteriors, axis=1)] = 1

        posteriors = posteriors * labels
        counts = np.sum(posteriors, 0)
        weighted_avg = np.dot(posteriors.T, x)

        means = weighted_avg / counts[:, np.newaxis]

        sigmas = np.zeros((self.expected_components, x.shape[1], x.shape[1]))
        for i in range(self.expected_components):
            difference = x - means[i, :]
            weighted_difference = posteriors[:, i][:, np.newaxis] * difference
            sigmas[i] = np.dot(weighted_difference.T, difference) / counts[i]

        priors = counts / len(x)

        return means, sigmas, priors

    def expectation_phase(self, x, means, sigmas, priors):
        expected_components, _ = means.shape
        likelihood = [multivariate_normal.pdf(x, means[i, :], sigmas[i, :, :], allow_singular=True)
                      for i in range(expected_components)]
        likelihood = np.asarray(likelihood).T
        num = np.asarray([likelihood[:, j] * priors[j] for j in range(expected_components)]).T
        denom = np.sum(num, 1)

        posteriors = np.asarray([num[:, j] / denom for j in range(expected_components)]).T

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

    def tissue_segmentation(self, T1, T2, brain_mask, type='knn', operation='EM'):
        t1_array = self.min_max_normalization(T1)
        t2_array = self.min_max_normalization(T2)

        t1_vector = t1_array[brain_mask == 255].flatten()
        t2_vector = t2_array[brain_mask == 255].flatten()

        data_vector = np.array(t1_vector)[:, np.newaxis]

        if operation == 'EM':
            number_samples = len(data_vector)
            labels = np.zeros((number_samples, self.expected_components))

            if type == 'knn':
                kmeans = KMeans(n_clusters=self.expected_components, random_state=self.seed).fit(data_vector)
                labels[np.arange(number_samples), kmeans.labels_] = 1
            elif type == 'random':
                rng = np.random.default_rng(seed=self.seed)
                idx = rng.choice(self.expected_components, size=number_samples)
                labels[np.arange(number_samples), idx] = 1

            means, sigmas = self.get_initial_values(data_vector, labels)

            priors = np.ones((self.expected_components, 1)) / self.expected_components

            previous_log_likelihood = 0

            start_time = time.time()

            for it in tqdm(range(self.maximum_iteration),desc=f'T1: Expectation Maximination in progress with {type} init'):
                posteriors, likelihood = self.expectation_phase(data_vector, means, sigmas, priors)

                for i in range(self.expected_components):
                    likelihood[:, i] = likelihood[:, i] * priors[i]

                log_likelihood = np.sum(np.log(np.sum(likelihood, 1)), 0)
                difference = abs(previous_log_likelihood - log_likelihood)
                previous_log_likelihood = log_likelihood

                if difference < self.change_tolerance:
                    break

                means, sigmas, priors = self.maximation_phase(data_vector, posteriors)

            posteriors, likelihood = self.expectation_phase(data_vector, means, sigmas, priors)

            predictions = np.argmax(posteriors, 1)

            predictions_image = brain_mask.flatten()
            predictions_image[predictions_image == 255] = predictions + 1

            t1_segmentation_result = predictions_image.reshape(T1.shape)

            t1_time = time.time() - start_time

        if operation == 'Kmeans':
            start_time = time.time()
            model = KMeans(n_clusters=self.expected_components, random_state=self.seed).fit(data_vector)
            predictions = model.predict(data_vector)
            predictions_image = brain_mask.flatten()
            predictions_image[predictions_image == 255] = predictions + 1
            t1_segmentation_result = predictions_image.reshape(T1.shape)
            t1_time = time.time() - start_time

        data_vector = np.array([t1_vector, t2_vector]).T

        if operation == 'EM':
            number_samples = len(data_vector)
            labels = np.zeros((number_samples, self.expected_components))

            if type == 'knn':
                kmeans = KMeans(n_clusters=self.expected_components, random_state=self.seed).fit(data_vector)
                labels[np.arange(number_samples), kmeans.labels_] = 1
            elif type == 'random':
                rng = np.random.default_rng(seed=self.seed)
                idx = rng.choice(self.expected_components, size=number_samples)
                labels[np.arange(number_samples), idx] = 1

            means, sigmas = self.get_initial_values(data_vector, labels)

            priors = np.ones((self.expected_components, 1)) / self.expected_components

            previous_log_likelihood = 0

            start_time = time.time()

            for it in tqdm(range(self.maximum_iteration),desc=f'T1+T2: Expectation Maximination in progress with {type} init'):
                posteriors, likelihood = self.expectation_phase(data_vector, means, sigmas, priors)

                for i in range(self.expected_components):
                    likelihood[:, i] = likelihood[:, i] * priors[i]

                log_likelihood = np.sum(np.log(np.sum(likelihood, 1)), 0)
                difference = abs(previous_log_likelihood - log_likelihood)
                previous_log_likelihood = log_likelihood

                if difference < self.change_tolerance:
                    break

                means, sigmas, priors = self.maximation_phase(data_vector, posteriors)

            posteriors, likelihood = self.expectation_phase(data_vector, means, sigmas, priors)

            predictions = np.argmax(posteriors, 1)

            predictions_image = brain_mask.flatten()
            predictions_image[predictions_image == 255] = predictions + 1

            t2_segmentation_result = predictions_image.reshape(T1.shape)

            t2_time = time.time() - start_time

        if operation == 'Kmeans':
            start_time = time.time()
            model = KMeans(n_clusters=self.expected_components, random_state=self.seed).fit(data_vector)
            predictions = model.predict(data_vector)
            predictions_image = brain_mask.flatten()
            predictions_image[predictions_image == 255] = predictions + 1
            t2_segmentation_result = predictions_image.reshape(T1.shape)
            t2_time = time.time() - start_time

        return t1_segmentation_result, t2_segmentation_result, t1_time, t2_time

    def min_max_normalization(self, image):
        image = (image - image.min()) / (image.max() - image.min()) * 255
        return image

    def match_pred_with_gt(self, prediction, ground_truth):
        wm = np.zeros_like(prediction)
        gm = np.zeros_like(prediction)
        cfs = np.zeros_like(prediction)

        for tets_prob in range(self.expected_components):
            probs = []
            for prob in range(self.expected_components):
                gt_layer = np.where(ground_truth == prob + 1, 1, 0)
                test = np.where(prediction == tets_prob + 1, 1, 0)
                intersection = np.sum(np.logical_and(gt_layer, test))
                union = np.sum(np.logical_or(gt_layer, test))
                iou = intersection / union
                probs.append(iou)

            max_location = probs.index(max(probs))

            if (max_location + 1) == 1:
                cfs[prediction == tets_prob + 1] = max_location + 1
            if (max_location + 1) == 2:
                gm[prediction == tets_prob + 1] = max_location + 1
            if (max_location + 1) == 3:
                wm[prediction == tets_prob + 1] = max_location + 1

        return wm, gm, cfs

    def dice_score(self, ground_truth, prediction):
        classes = np.unique(ground_truth[ground_truth != 0])
        dice = np.zeros((len(classes)))
        for i in classes:
            binary_prediction = np.where(prediction == i, 1, 0)
            binary_ground_truth = np.where(ground_truth == i, 1, 0)
            dice[i - 1] = np.sum(binary_prediction[binary_ground_truth == 1]) * 2.0 / (
                    np.sum(binary_prediction) + np.sum(binary_prediction))
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
