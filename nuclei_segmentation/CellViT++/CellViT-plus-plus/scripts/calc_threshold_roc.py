# -*- coding: utf-8 -*-
# Find best classifier threshold using ROC curve
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
from torchmetrics import F1Score, AUROC, Accuracy, ROC

from pathlib import Path

logdir = Path("path/to/logdir")

val_result_dir = logdir / "val_results"

gt = torch.load(val_result_dir / "gt.pt")
probabilities = torch.load(val_result_dir / "probabilities.pt")

roc_function = ROC(task="binary")
f1_score_func = F1Score(task="binary")
auroc_func = AUROC(task="binary")
accuracy_func = Accuracy(task="binary")

fpr, tpr, thresholds = roc_function(probabilities[:, 1], gt)
thresh = thresholds[torch.argmax(tpr - fpr)]

# Find the threshold that maximizes F1 score
pred_argmax = probabilities[:, 1] > 0.5
pred_thresh = probabilities[:, 1] > thresh


f1_argmax = f1_score_func(pred_argmax, gt)
f1_thresh = f1_score_func(pred_thresh, gt)
acc_argmax = accuracy_func(pred_argmax, gt)
acc_thresh = accuracy_func(pred_thresh, gt)
auroc = auroc_func(probabilities[:, 1], gt)

print(f"ROC AUC: {auroc}")
print(f"F1 argmax: {f1_argmax}")
print(f"F1 thresh: {f1_thresh}")
print(f"Acc argmax: {acc_argmax}")
print(f"Acc thresh: {acc_thresh}")
print(f"Thresholds: {thresh}")
