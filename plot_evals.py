import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle

for time in [3, 6, 12, 18, 24, 32, 40, 48]:

    # Load JSON data
    with open(f'mbtcn_evals/{time}_death_time_adam.json', 'r') as file:
        data = json.load(file)

    # Extract predictions and true labels
    all_targets = np.array(data['all_targets'])
    all_preds = np.array(data['all_preds'])

    to_bin = {
        0: [1, 2]
    }

    # Number of classes (assumes all_targets is one-hot encoded)
    n_classes = all_targets.shape[1] - len(np.array(to_bin.values()).flatten().tolist()) - len(to_bin.keys())

    for key, val in to_bin.items():
        all_targets[:, val] = key

    print(all_targets)

    exit()

    # Initializing dictionaries for ROC and PRC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_targets[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(all_targets[:, i], all_preds[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Plot ROC curves for all classes
    colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'black'])
    plt.figure(figsize=(12, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC')
    plt.legend(loc="lower right")
    plt.savefig(f'mbtcn_graphs/{time}_death_time_adam_ROC.png')

    # Plot Precision-Recall curves for all classes
    plt.figure(figsize=(12, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'Precision-Recall curve of class {i} (area = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-Class Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'mbtcn_graphs/{time}_death_time_adam_PRC.png')
