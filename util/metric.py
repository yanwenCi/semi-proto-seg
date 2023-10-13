"""
Metrics for computing evalutation results
"""

import numpy as np
import skimage.measure as measure
import skimage.morphology as morphology

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_runs:
            number of test runs
    """
    def __init__(self, max_label=20, n_runs=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_runs = 1 if n_runs is None else n_runs

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]

    def record(self, pred, target, labels=None, n_run=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape

        if self.n_runs == 1:
            n_run = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            if target_idx_j:  # if ground-truth contains this class
                tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
                fp_arr[label] = len(pred_idx_j - target_idx_j)
                fn_arr[label] = len(target_idx_j - pred_idx_j)

        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)

    def get_dice(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.vstack(self.tp_lst[run])
                      for run in range(self.n_runs)]
            fp_sum = [np.vstack(self.fp_lst[run])
                      for run in range(self.n_runs)]
            fn_sum = [np.vstack(self.fn_lst[run])
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([np.mean(2*tp_sum[run] / (2*tp_sum[run] + fp_sum[run] + fn_sum[run]),axis=0).take(labels)
                                    for run in range(self.n_runs)])
            mIoU = np.mean(mIoU_class, axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.vstack(self.tp_lst[n_run])
            fp_sum = np.vstack(self.fp_lst[n_run])
            fn_sum = np.vstack(self.fn_lst[n_run])

            # Compute mean IoU classwisely and average over classes
            mIoU_class = np.nanmean((2*tp_sum / (2*tp_sum + fp_sum + fn_sum)), axis=0).take(labels)
            #mIoU = mIoU_class.mean()
            mIoU_std = np.nanstd((2*tp_sum / (2*tp_sum + fp_sum + fn_sum)), axis=0).take(labels)
            return mIoU_class, mIoU_std

    def get_dice_batch(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([2*tp_sum[run] / (2*tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (np.nanmean(mIoU_class, axis=0), np.nanstd(mIoU_class, axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.sum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.sum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.sum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = 2*tp_sum / (2*tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU



    def get_mIoU(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mIoU_binary(self, n_run=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0)
                      for run in range(self.n_runs)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])]
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU





def positive_lesion_rate(tgt, y_last, thre, direct='gt'):
    #tgt is the perspective direction, it is always first: 3d np array
    tgt=tgt.astype(np.bool)
    y_last=y_last.astype(np.bool)
    tgt=morphology.remove_small_holes(morphology.remove_small_objects(tgt,4, connectivity=2),4,connectivity=2)
    y_last=morphology.remove_small_holes(morphology.remove_small_objects(y_last,4, connectivity=2),4,connectivity=2)

    label_image, number_of_labels=measure.label(tgt, background = 0, return_num = True, connectivity=1)
    measures=np.empty(number_of_labels)
    i=0
    th_count=[0]*(len(thre))
    number_of_lesions_=number_of_labels+0
    for j,region in zip(range(number_of_labels),measure.regionprops(label_image, intensity_image=tgt)):
        lesion=np.zeros(tgt.shape)
        if region.area<4:
            measures=measures[:-1]
            number_of_lesions_=number_of_lesions_-1
        else:
            #lesion[minx:maxx, miny:maxy, minz:maxz]=label_image[minx:maxx, miny:maxy, minz:maxz]
            lesion[label_image==j+1]=1
            #print(label_image.min(), label_image.max())
            measure1=calculate_lesion(lesion, y_last)
            #measure1,measure2, measure3=calculate_pn(img[minx:maxx, miny:maxy, minz:maxz], tgt[minx:maxx, miny:maxy, minz:maxz])
            measures[i]=measure1
            i+=1
            for k, th in enumerate(thre):
                if measure1>th:
                    th_count[k]+=1

    return measures, th_count, number_of_lesions_

def calculate_lesion(y_first, y_last):
    # first is always the denominator
    # fist  is considered as lesion pred, then for that lesion, fp+tp=lesion
    y_first=y_first.reshape(1,-1)
    y_last = y_last.reshape(1, -1)        
    
    tp = np.sum(y_last * y_first, axis=-1)
    #lesion_sum=np.sum(y_pred, axis=-1)
    #tn = np.sum((1 - y_last) * (1 - y_first), axis=-1)
    fp = np.sum((1 - y_last) * y_first, axis=-1)
    #fn = np.sum(y_true * (1 - y_pred), axis=-1)
    return tp/(tp+fp)
