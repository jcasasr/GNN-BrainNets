import csv
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import logging
from datetime import datetime
import random
from typing import List

from Subject import Subject
from gnns.GCN import GCN
from gnns.GAT import GAT
from gnns.GIN import GIN
from gnns.ChebNet import ChebNet
from gnns.GraphSAGE import GraphSAGE
from gnns.FPGCN import FPGCN
import utils.utils as utils
import utils.gnn as gnn
import utils.score as score
import utils.early_stopping as EarlyStopping
from utils.timer import Timer
from config import *


##############
### PARAMS ###
##############
# Debug mode and plotting
DEBUG = False
PLOTS = False
# Number of classes:
# binary: 2 
# multiclass: 4
NUM_CLASSES = 4
# Experiment settings
N_ITERATIONS = 4
NUM_EPOCHS = 200
N_SPLITS = 4
# Early stopping parameters
PATIENCE = 10
DELTA = 0.01
# Real dataset (no use "*.pkl" to avoid loading synthetic subjects for testing)
SUBJECT_COHORT_LIST = ["ALL", "BCN", "NAP"]
SUBJECT_PATTERNS_LIST = [["BCN-*.pkl", "NAP-*.pkl"], ["BCN-*.pkl"], ["NAP-*.pkl"]] #  (do not use "*.pkl" to avoid loading synthetic subjects for testing)
# Device configuration
device = torch.device(get_device("cpu"))

###################
### Data models ###
###################
# Data types to be tested
# Tuple: (name, graph_structure, [embeddings])
# - FA: ("FA", "FA", ["FA"])
# - GM: ("GM", "GM", ["GM"])
# - RS: ("RS", "RS", ["RS"])
# - Multi-layer: ("ML", "ML", ["ML"])
# - Multi-modal: ("MM", "FA", ["GM", "RS"])
TEST_METHODS = [("MM", "FA", ["FA", "GM", "RS"]),
                ("MM-THR", "FA-THR", ["FA-THR", "GM-THR", "RS-THR"])]
# List of metrics to be used as embeddings for GNNs
METRIC_LIST = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality', 
               'Clustering', 'EigenvectorCentrality', 'PageRank']

##################
### GNN params ###
##################
# Tuple: ("model_name", [dim_h1, dim_h2])
MODEL_ARCHITECTURES = [("GCN", [32, 0]),
                       ("GCN", [32, 16]),
                       ("GAT", [32, 0]),
                       ("GAT", [32, 16]),
                       ("GIN", [32, 3]),
                       ("GIN", [32, 5]),
                       ("Cheb", [32, 3]),
                       ("Cheb", [32, 5]),
                       ("SAGE", [32, 3]),
                       ("SAGE", [32, 5]),
                       ("FPGCN", [1, 0]),
                       ("FPGCN", [2, 0])]
LEARNING_RATES = [0.01, 0.001, 0.0001]
WEIGHT_DECAYS = [5e-4, 5e-6]
# GAT params
NUM_HEADS = 4
DROPOUT = 0.6
# Data augmentation
DATA_AUGMENTATIONS = [False, True]
DA_PATTERN = "SYN-*.pkl"


if __name__ == "__main__":
    # Timer
    glb_timer = Timer()
    # create a list to store results
    global_results = []

    # for each subject cohort
    for SUBJECT_COHORT, SUBJECT_PATTERNS in zip(SUBJECT_COHORT_LIST, SUBJECT_PATTERNS_LIST):
    
        # for each data type
        for DATA_TYPE in TEST_METHODS:
            # name
            data_type_name = DATA_TYPE[0]
            # graph structure (topology)
            data_type_structure = DATA_TYPE[1]
            # embbedings
            data_type_embbedings = DATA_TYPE[2]

            # for each model architecture
            for model_arch in MODEL_ARCHITECTURES:
                MODEL = model_arch[0]
                # DIM_IN is the number of embeddings / node
                DIM_IN = len(METRIC_LIST) * len(data_type_embbedings)
                # DIM_OUT is the number of classes
                DIM_OUT = NUM_CLASSES
                # Hidden layers
                DIM_H1 = model_arch[1][0]
                DIM_H2 = model_arch[1][1]

                # for each Learning Rate and Weight Decay
                for LEARNING_RATE in LEARNING_RATES:
                    # for each weight decay
                    for WEIGHT_DECAY in WEIGHT_DECAYS:
                        # with and without data augmentation
                        for data_augmentation in DATA_AUGMENTATIONS:
                            # Local timer
                            lcl_timer = Timer()
                            # Set logger
                            EXP_NAME = "Step-6-2_"+ data_type_name +"_"+ MODEL +"-"+ str(DIM_H1) +"-"+ str(DIM_H2) +"_lr"+ str(LEARNING_RATE) +"_wd"+ str(WEIGHT_DECAY) +("_DA" if data_augmentation else "_noDA")
                            TIME_NAME = datetime.today().strftime("%Y%m%d-%H%M%S")
                            LOGGING_NAME = TIME_NAME +"_"+ EXP_NAME
                            logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

                            logger.info(f"====================================================")
                            logger.info(f"=== EXPERIMENT  : {EXP_NAME}")
                            logger.info(f"=== COHORT      : {SUBJECT_COHORT} ('{SUBJECT_PATTERNS}')")
                            logger.info(f"=== DATA_AUGMENT: {data_augmentation} (Pattern: '{DA_PATTERN}')")
                            logger.info(f"=== DATA_TYPE   : {data_type_name} (Structure: '{data_type_structure}' | Embeddings: {data_type_embbedings})")
                            logger.info(f"=== METRICS     : {METRIC_LIST}")
                            logger.info(f"=== MODEL ARCH. : {model_arch}")
                            logger.info(f"=== DIMENSIONS  : DIM_IN: {DIM_IN} | DIM_OUT: {DIM_OUT} | DIM_H1: {DIM_H1} | DIM_H2: {DIM_H2}")
                            logger.info(f"=== HYPERPARAMS : LR: {LEARNING_RATE} | WD: {WEIGHT_DECAY}")
                            logger.info(f"=== TRAINING    : ITERATIONS: {N_ITERATIONS} | N_SPLITS: {N_SPLITS} | NUM_EPOCHS: {NUM_EPOCHS}")
                            logger.info(f"=== DEVICE      : {device}")
                            logger.info(f"====================================================\n")
                            
                            # load data
                            subjects = []
                            for subject_pattern in SUBJECT_PATTERNS:
                                subjects += utils.load_subjects(path_input=PATH, subject_pattern=subject_pattern, debug=DEBUG)
                            
                            target = [int(subject.get_mstype(type="multiclass")) for subject in subjects]
                            logger.info("MS Types: {}".format(sorted(np.unique(target))))
                            # Matrix for prediction
                            preds = np.zeros(len(subjects), dtype=int)

                            # If data augmentation is enabled, load synthetic subjects
                            if data_augmentation:
                                logger.info("*** USING Data Augmentation Strategy: VAE model")
                                
                                # Get all SYN subjects
                                subjects_syn = utils.load_subjects(path_input=PATH, subject_pattern=DA_PATTERN, debug=DEBUG)
                                logger.info("   Loaded {} SYN subjects".format(len(subjects_syn)))
                                
                                # Filter subjects generated from the same cohort
                                # TODO: Uncomment the following lines if you want to filter synthetic subjects by cohort. 
                                # All synthetic subjects are generated from the VAE of both cohorts (BCN and NAP), so if you want to use only synthetic subjects 
                                # from the same cohort as the real subjects, you must generate them from the VAE of each cohort separately.
                                #subjects_syn = [x for x in subjects_syn if x.get_cohort() in SUBJECT_COHORT]
                                #logger.info("   Filtering SYN subjects generated from {} cohorts: remaining {} subjects".format(SUBJECT_COHORT, len(subjects_syn)))

                                target_syn = [int(x.get_mstype(type="multiclass")) for x in subjects_syn]
                                logger.info("   MS Types: {}".format(sorted(np.unique(target_syn))))
                            else:
                                logger.info("*** NO Data Augmentation Strategy")
                                subjects_syn = []
                                target_syn = []

                            # For each iteration
                            glb_auc_roc_macro = []
                            glb_auc_roc_micro = []
                            glb_auc_pr = []
                            glb_acc = []
                            glb_f1 = []

                            for iteration in range(N_ITERATIONS):
                                logger.info("++++ Iteration: {} / {}".format(iteration+1, N_ITERATIONS))

                                # k-fold cross-validation
                                skf = StratifiedKFold(n_splits=N_SPLITS)
                                fold = 0
                                for train_index, test_index in skf.split(subjects, target):
                                    fold += 1
                                    logging.info("Fold: {}".format(fold))

                                    # split dataset
                                    X_train = [subjects[i] for i in train_index]
                                    X_test  = [subjects[i] for i in test_index]
                                    y_train, y_test = np.array(target)[train_index], np.array(target)[test_index]

                                    # If data augmentation is enabled, concatenate the synthetic subjects to the training set
                                    if data_augmentation:
                                        # Concatenate train values
                                        X_train = X_train + subjects_syn
                                        y_train = np.concatenate((y_train, np.array(target_syn)))
                                        # Shuffle training samples
                                        index_shuf = list(range(len(X_train)))
                                        random.shuffle(index_shuf)
                                        X_train_sh = []
                                        y_train_sh = []
                                        for i in index_shuf:
                                            X_train_sh.append(X_train[i])
                                            y_train_sh.append(y_train[i])
                                        X_train = X_train_sh
                                        y_train = np.array(y_train_sh)
                                    
                                    prop_train = np.where(y_train >= 1)[0].shape[0] / y_train.shape[0]
                                    prop_test = np.where(y_test >= 1)[0].shape[0] / y_test.shape[0]
                                    logger.info("Train set size     : {}".format(len(X_train)))
                                    logger.info("Test set size      : {}".format(len(X_test)))
                                    logger.info("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
                                    logger.info("Test set % of pwMS : {:.4f} ({})".format(prop_test, y_test.sum()))

                                    # list of Data structures (one for each subject)
                                    train_graphs = []
                                    for i in range(len(X_train)):
                                        g = gnn.array_to_graph(subject=X_train[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                                    num_classes=NUM_CLASSES, thr=THR[data_type_structure], metric_list=METRIC_LIST, logger=logger)
                                        train_graphs.append(g)
                                    gnn.print_graphs_info(graphs=train_graphs, name="Train", thr=THR[data_type_structure], logger=logger)
                                        
                                    # Test dataset
                                    test_graphs = []
                                    for i in range(len(X_test)):
                                        g = gnn.array_to_graph(subject=X_test[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                                    num_classes=NUM_CLASSES, thr=THR[data_type_structure], metric_list=METRIC_LIST, logger=logger)
                                        test_graphs.append(g)
                                    gnn.print_graphs_info(graphs=test_graphs, name="Test", thr=THR[data_type_structure], logger=logger)

                                    # create the model
                                    if MODEL == "GCN":
                                        model = GCN(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                                    patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                                    elif MODEL == "GAT":
                                        model = GAT(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, heads=NUM_HEADS, dropout=DROPOUT, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                                    patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                                    elif MODEL == "GIN":
                                        model = GIN(dim_in=DIM_IN, dim_out=DIM_OUT, hidden=DIM_H1, num_layers=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                                    patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                                    elif MODEL == "Cheb":
                                        model = ChebNet(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, k=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                                        patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                                    elif MODEL == "SAGE":
                                        model = GraphSAGE(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, num_layers=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                                        patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                                    elif MODEL == "FPGCN":
                                        model = FPGCN(dim_in=DIM_IN, dim_out=DIM_OUT, num_layers=DIM_H1, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                                    patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                                    else:
                                        logger.error("Model not implemented!")
                                        raise NotImplementedError("Model not implemented!")

                                    # Train the model
                                    model.fit(num_epocs=NUM_EPOCHS, train_graphs=train_graphs)
                                    y_pred = model.test(test_graphs=test_graphs, y_test=y_test)
                                    # Save predictions to compute final test scores
                                    preds[test_index] = y_pred

                                # Compute metrics
                                auc_roc_macro, auc_roc_micro, auc_pr, acc, _, f1, conf_matrix = score.compute_auc_multiclass(target=target, preds=preds, logger=logger)
                                if PLOTS:
                                    score.plot_auc_roc_multiclass(target=target, preds=preds, name_out=LOGGING_NAME +"_"+ str(iteration+1))
                                
                                # Add to global results
                                glb_auc_roc_macro.append(auc_roc_macro)
                                glb_auc_roc_micro.append(auc_roc_micro)
                                glb_auc_pr.append(auc_pr)
                                glb_acc.append(acc)
                                glb_f1.append(f1)

                            # Final scores
                            
                            logger.info("*** Final AUC ROC score (macro): {:.4f} +- {:.4f}".format(np.mean(glb_auc_roc_macro), np.std(glb_auc_roc_macro)))
                            logger.info(".   AUC ROC scores (macro)     : {}".format(glb_auc_roc_macro))
                            logger.info("*** Final AUC ROC score (micro): {:.4f} +- {:.4f}".format(np.mean(glb_auc_roc_micro), np.std(glb_auc_roc_micro)))
                            logger.info(".   AUC ROC scores (micro)     : {}".format(glb_auc_roc_micro))
                            logger.info("*** Final AUC PR score: {:.4f} +- {:.4f}".format(np.mean(glb_auc_pr), np.std(glb_auc_pr)))
                            logger.info(".   AUC PR scores     : {}".format(glb_auc_pr))
                            logger.info("*** Final ACC score: {:.4f} +- {:.4f}".format(np.mean(glb_acc), np.std(glb_acc)))
                            logger.info(".   ACC scores        : {}".format(glb_acc))
                            logger.info("*** Final F1 score: {:.4f} +- {:.4f}".format(np.mean(glb_f1), np.std(glb_f1)))
                            logger.info(".   F1 scores         : {}".format(glb_f1))
                            logger.info("*** PROCESS FINISHED! [Running time = {}]\n".format(lcl_timer))

                            # Store to results
                            global_results.append({"NAME": EXP_NAME,
                                                "Data_type": data_type_name,
                                                "Data_structure": data_type_structure,
                                                "Data_embbedings": data_type_embbedings,
                                                "DA": data_augmentation,
                                                "Model": MODEL,
                                                "Dim_in": DIM_IN,
                                                "Dim_out": DIM_OUT,
                                                "Dim_h1": DIM_H1,
                                                "Dim_h2": DIM_H2,
                                                "Learning_rate": LEARNING_RATE,
                                                "Weight_decay": WEIGHT_DECAY,
                                                "AUC_ROC_MACRO_AVE": np.mean(glb_auc_roc_macro),
                                                "AUC_ROC_MACRO_STD": np.std(glb_auc_roc_macro),
                                                "AUC_ROC_MICRO_AVE": np.mean(glb_auc_roc_micro),
                                                "AUC_ROC_MICRO_STD": np.std(glb_auc_roc_micro),
                                                "AUC_PR_AVE": np.mean(glb_auc_pr),
                                                "AUC_PR_STD": np.std(glb_auc_pr),
                                                "ACC_AVE": np.mean(glb_acc),
                                                "ACC_STD": np.std(glb_acc),
                                                "F1_AVE": np.mean(glb_f1),
                                                "F1_STD": np.std(glb_f1)})

            # Export to TXT file
            with open("results/"+ TIME_NAME +"_GNNs_MCL_"+ SUBJECT_COHORT +"_"+ data_type_name +".txt", "w") as f:
                for line in global_results:
                    f.write("{} : AUC ROC (macro) = {:.4f} +- {:.4f} ; AUC ROC (micro) = {:.4f} +- {:.4f} ; AUC PR = {:.4f} +- {:.4f} ; ACC = {:.4f} +- {:.4f} ; F1 = {:.4f} +- {:.4f} \n".format(
                        line["NAME"], line["AUC_ROC_MACRO_AVE"], line["AUC_ROC_MACRO_STD"], line["AUC_ROC_MICRO_AVE"], line["AUC_ROC_MICRO_STD"], line["AUC_PR_AVE"], line["AUC_PR_STD"], line["ACC_AVE"], line["ACC_STD"], line["F1_AVE"], line["F1_STD"]))

            # Export to CSV file
            with open("results/"+ TIME_NAME +"_GNNs_MCL_"+ SUBJECT_COHORT +"_"+ data_type_name +".csv", "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=global_results[0].keys())
                writer.writeheader()
                writer.writerows(global_results)

            # reset the list to store results
            global_results = []

    logger.info("### PROCESS FINISHED! [Total running time = {}]\n".format(glb_timer))