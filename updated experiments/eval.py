import os
import random
import shutil
from pathlib import Path
import json
import math

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd


class Metrics:
    def __init__(self, labels, scores, num_metrics=3, best_threshold=False):
        ## REAL = 0, FAKE = 1
        self.labels = labels
        self.scores = scores
        self.num_metrics = num_metrics
        self.best_threshold = best_threshold

    def computation(self):
        if self.num_metrics == 3:
            fpr, tpr, thresholds = roc_curve(self.labels, self.scores)
            idx = np.where(tpr >= 0.95)[0][0]
            fpr_95 = fpr[idx]

            roc_auc = roc_auc_score(self.labels, self.scores)
            precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
            pr_auc = auc(recall, precision)

            if self.best_threshold:

                distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
                best_threshold = thresholds[np.argmin(distances)]
                print("Best threshold(ROC):", best_threshold)
                return roc_auc, pr_auc, fpr_95, best_threshold

            else:
                return roc_auc, pr_auc, fpr_95

        else:
            print("Precise your target.")
