import numpy as np
from skimage import measure, morphology


def true_positive_lesion(tgt, y_last, thresholds, direction='gt'):
    """
    Computes per-lesion positive detection rate (TP / (TP + FP)) for predicted lesions in a 3D binary mask.

    Parameters:
        tgt (np.ndarray): Ground truth binary mask (3D).
        y_last (np.ndarray): Predicted binary mask (3D).
        thresholds (list of float): List of thresholds to count how many lesions exceed each threshold.
        direction (str): Unused placeholder. Default is 'gt'.

    Returns:
        lesion_scores (np.ndarray): Array of scores (TP / (TP + FP)) for each lesion in the ground truth.
        threshold_counts (list of int): Number of lesions whose score exceeds each threshold.
        total_lesions (int): Total number of valid lesions (after removing small ones).
    """
    # Ensure binary input
    tgt = tgt.astype(bool)
    y_last = y_last.astype(bool)

    # Clean small regions
    tgt = morphology.remove_small_objects(tgt, min_size=4, connectivity=2)
    tgt = morphology.remove_small_holes(tgt, area_threshold=4, connectivity=2)

    y_last = morphology.remove_small_objects(y_last, min_size=4, connectivity=2)
    y_last = morphology.remove_small_holes(y_last, area_threshold=4, connectivity=2)

    # Label connected components in the ground truth
    label_image, num_labels = measure.label(tgt, background=0, return_num=True, connectivity=1)

    lesion_scores = []
    threshold_counts = [0] * len(thresholds)

    for region in measure.regionprops(label_image, intensity_image=tgt):
        if region.area < 4:
            continue  # Skip small lesions

        lesion_mask = np.zeros_like(tgt, dtype=bool)
        lesion_mask[label_image == region.label] = True

        score = calculate_lesion_score(lesion_mask, y_last)
        lesion_scores.append(score)

        for i, th in enumerate(thresholds):
            if score > th:
                threshold_counts[i] += 1

    lesion_scores = np.array(lesion_scores)
    total_lesions = len(lesion_scores)

    return lesion_scores, threshold_counts, total_lesions


def calculate_lesion_score(lesion_mask, prediction_mask):
    """
    Calculates precision for a lesion: TP / (TP + FP)

    Parameters:
        lesion_mask (np.ndarray): Binary mask for a single lesion.
        prediction_mask (np.ndarray): Full predicted binary mask.

    Returns:
        score (float): Precision for the lesion region.
    """
    lesion_flat = lesion_mask.ravel().astype(int)
    prediction_flat = prediction_mask.ravel().astype(int)

    tp = np.sum(prediction_flat * lesion_flat)
    fp = np.sum((1 - lesion_flat) * prediction_flat)

    if tp + fp == 0:
        return 0.0  # Avoid division by zero

    return tp / (tp + fp)
