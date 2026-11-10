import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch import from_numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt


####
# Binary classification
###
def compute_auc_binary(target, preds, logger=None):
    '''
    Compute AUC-ROC and AUC-PR for binary classification
    Args:
        target (list): List of true labels
        preds (list): List of predicted probabilities
        logger (logging.Logger): Logger object for logging
    '''
    # Compute metrics
    auc_roc = roc_auc_score(target, preds)
    auc_pr = average_precision_score(target, preds)
    prop = np.sum(target) / len(target)
        
    # Compute best threshold
    best_acc = 0
    best_th = 0
    for th in preds:
        acc = accuracy_score(target, (preds >= th).astype(int))
        if acc >= best_acc:
            best_acc = acc
            best_th = th
    # Compute confusion matrix
    conf_matrix = confusion_matrix(target, (preds >= best_th).astype(int))
    f1 = f1_score(target, (preds >= best_th).astype(int), average='binary')

    logger.info("")
    logger.info("AUC ROC  : {:.4f}".format(auc_roc))
    logger.info("AUC PR   : {:.4f}".format(auc_pr))
    logger.info("ACC      : {:.4f} (threshold = {})".format(best_acc, best_th))
    logger.info("F1 Score : {:.4f}".format(f1))
    logger.info("% of pwMS: {:.4f}".format(prop))
    logger.info("Confusion matrix: \n{}".format(conf_matrix))

    return auc_roc, auc_pr, best_acc, best_th, f1, conf_matrix


def plot_auc_roc_binary(target, preds, name_out:str):
    '''
    Plot AUC-ROC and AUC-PR for binary classification
    Args:
        target (list): List of true labels
        preds (list): List of predicted probabilities
        name_out (str): Output file name
    '''
    # Matplotlib settings
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['font.size'] = 14
    lw = 2

    prop = np.sum(target) / len(target)

    # compute AUC-ROC and ROC curve
    auc_roc = roc_auc_score(target, preds)
    fpr, tpr, _ = roc_curve(target, preds)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=lw)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (AUC-ROC: {:.4f})".format(auc_roc))
    plt.savefig(os.path.join("figs", name_out +"_ROC.png"), dpi=300, bbox_inches='tight', pad_inches=0)

    # Compute AUC-PR
    auc_pr = average_precision_score(target, preds)
    prec, recall, _ = precision_recall_curve(target, preds)

    plt.figure()
    plt.plot(recall, prec, color="darkorange", lw=lw)
    plt.plot([0, 1], [prop, prop], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve (AUC-PR: {:.4f})".format(auc_pr))
    plt.savefig(os.path.join("figs", name_out +"_P-R.png"), dpi=300, bbox_inches='tight', pad_inches=0)


###
# Multiclass classification
###
def compute_auc_multiclass(target, preds, logger=None, num_classes:int=4):
    # Compute One Hot Encoding
    target_oh = F.one_hot(from_numpy(np.array(target)), num_classes=num_classes)
    preds_oh = F.one_hot(from_numpy(preds), num_classes=num_classes)
    
    # Compute metrics
    auc_roc_macro = roc_auc_score(target_oh, preds_oh, average='macro', multi_class='ovr')
    auc_roc_micro = roc_auc_score(target_oh, preds_oh, average='micro', multi_class='ovr')
    auc_pr = average_precision_score(target_oh, preds_oh)
    # Compute ACC
    acc = accuracy_score(target, preds)
    # Compute confusion matrix
    conf_matrix = confusion_matrix(target, preds)
    f1 = f1_score(target_oh, preds_oh, average='weighted')

    logger.info("")
    logger.info("AUC ROC  : {:.4f}".format(auc_roc_macro))
    logger.info("AUC PR   : {:.4f}".format(auc_pr))
    logger.info("ACC      : {:.4f}".format(acc))
    logger.info("F1 Score : {:.4f}".format(f1))
    logger.info("% of HV  : {:.4f}".format(np.sum([i == 0 for i in target]) / len(target)))
    logger.info("% of pwMS: {:.4f}".format(np.sum([i >= 1 for i in target]) / len(target)))
    logger.info(".    MS=1: {:.4f}".format(np.sum([i == 1 for i in target]) / len(target)))
    logger.info(".    MS=2: {:.4f}".format(np.sum([i == 2 for i in target]) / len(target)))
    logger.info(".    MS=3: {:.4f}".format(np.sum([i == 3 for i in target]) / len(target)))
    logger.info("Confusion matrix: \n{}".format(conf_matrix))

    return auc_roc_macro, auc_roc_micro, auc_pr, acc, None, f1, conf_matrix


def plot_auc_roc_multiclass(target, preds, name_out:str):
    pass
