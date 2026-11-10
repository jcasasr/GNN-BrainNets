import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import random

from Subject import Subject
from utils.timer import Timer
import utils.utils as utils
import utils.graphs as graphs
import utils.gnn as gnn
import utils.score as score
from utils.vae import *
from gnns.GCN import GCN
from gnns.GAT import GAT


##################################
### Paths and general settings ###
##################################
DEBUG = True
PATH_BCN = "/Users/jcasasr/Library/CloudStorage/OneDrive-ComputerVisionCenter/Datasets-MRI-Graph/CVC_MRI-Data/BCN_output"
PATH_NAP = "/Users/jcasasr/Library/CloudStorage/OneDrive-ComputerVisionCenter/Datasets-MRI-Graph/CVC_MRI-Data/NAP_output"
PATH = "./data"
MATRIX_TYPES_ORIGINAL = ["FA", "GM", "RS"]
MATRIX_TYPES = ["FA", "GM", "RS", "ML"] # Types of valid matrices
NUM_NODES = 76

############################
#### GENERAL PARAMETERS ####
############################
# Cohorts and patterns
COHORTS = ["BCN", "NAP", "ALL"]
COHORTS_PATTERN = ["BCN-*.pkl", "NAP-*.pkl", "*.pkl"]
# input dimension (number of features)
INPUT_DIM = 2850
# Device configuration
device = "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

#################################
#### STEP 1. CREATE SUBJECTS ####
#################################
STEP1_SUBJECTS = True
# Number of synthetic samples to generate / ms_type
# [HV, MS type 1, MS type 2, MS type 3]
NUM_SAMPLES = {0: 50, 1: 50, 2: 5, 3: 5} 
# Scale of the random noise added to the mean latent vector
RANDOM_SCALES = [0.01, 0.1, 1.0, 5.0, 10.0, 100.0]
# ID of the new synthetic subjects  
COHORT_NEW = "SYN"
SYN_PATTERN = "SYN-*.pkl"
# Threshold for each matrix
THR = {'FA': 0.45, 'GM': 0.575, 'RS': 0.1}

############################
#### STEP 2. SL METRICS ####
############################
STEP2_SL_METRICS = True
METRIC_LIST = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality']#, 'Clustering', 'EigenvectorCentrality', 'PageRank']

#####################
#### STEP 3. GNN ####
#####################
STEP3_GNN = True
TEST_METHODS = [("MM", "FA", ["FA", "GM", "RS"])]
NUM_CLASSES = 2
N_ITERATIONS = 4
NUM_EPOCHS = 100
N_SPLITS = 4
NUM_HEADS = 4
DROPOUT = 0.6
# Model architecture
# Tuple: (model, [dim_h1, dim_h2])
MODEL_ARCHITECTURES = [("GCN", [32, 0])]
LEARNING_RATES = [0.001]
WEIGHT_DECAYS = [5e-6]


if __name__ == "__main__":
    # For each random scale
    for RANDOM_SCALE in RANDOM_SCALES:
        # global vars
        glb_acc_diff = []

        # For each COHORT
        for cohort, subject_pattern in zip(COHORTS, COHORTS_PATTERN):
            # Set logger
            LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_TEST_VAE-full_"+ cohort
            logger = utils.set_logging(logging_level=logging.INFO, logging_name=LOGGING_NAME)

            logger.info(f"====================================================")
            logger.info(f"=== COHORT      : {cohort} ('{subject_pattern}')")
            logger.info(f"=== NUM_SAMPLES : {NUM_SAMPLES}")
            logger.info(f"=== RANDOM_SCALE: {RANDOM_SCALE}")
            logger.info(f"=== DEVICE      : {device}")
            logger.info(f"====================================================\n")

            # Timer
            glb_timer = Timer()

            if STEP1_SUBJECTS:
                logger.info("### Step 1: generating subjects started!\n")
                step_timer = Timer()
                # Subject index
                subj_idx = 0

                # For each MS type
                for ms_type in [0, 1, 2, 3]:
                    num_samples = NUM_SAMPLES[ms_type]
                    logger.info(f"=== Creating {num_samples} subjects for {cohort} ('{subject_pattern}') and ms_type={ms_type} by using VAE ===")
                    
                    # Create FA matrices
                    subjects_FA = get_data_from_mstype("FA", subject_pattern, ms_type, logger)
                    matrices_FA = create_samples_from_vae(cohort, "FA", subjects_FA, num_samples, RANDOM_SCALE, logger)

                    # Create GM matrices
                    subjects_GM = get_data_from_mstype("GM", subject_pattern, ms_type, logger)
                    matrices_GM = create_samples_from_vae(cohort, "GM", subjects_GM, num_samples, RANDOM_SCALE, logger)

                    # Create RS matrices
                    subjects_RS = get_data_from_mstype("RS", subject_pattern, ms_type, logger)
                    matrices_RS = create_samples_from_vae(cohort, "RS", subjects_RS, num_samples, RANDOM_SCALE, logger)

                    # Join matrices and create the subjects
                    for i in range(num_samples):
                        # change the ID to "cohort-ID"
                        ID = COHORT_NEW + "-" + "{:04}".format(subj_idx)
                        logger.info("Creating subject {}...".format(ID))

                        # Create Subject object
                        subject = Subject(ID, cohort, "", ms_type)

                        # Convert flatten arrays to adjacency matrices
                        tmp_data_FA = get_adj_matrix_from_flattened(matrices_FA[i])
                        tmp_data_GM = get_adj_matrix_from_flattened(matrices_GM[i])
                        tmp_data_RS = get_adj_matrix_from_flattened(matrices_RS[i])
                        
                        # Set matrices
                        subject.set_matrix("FA", tmp_data_FA) # don't threshold FA matrix: thr=THR["FA"]
                        subject.set_matrix("GM", tmp_data_GM) # idem
                        subject.set_matrix("RS", tmp_data_RS) # idem

                        # print subject
                        logger.info(str(subject))

                        # Export subject
                        utils.save_subject(subject)
                        subj_idx += 1
                
                logger.info("### Step 1: finished with {}\n".format(step_timer))
            else:
                logger.warning("### Step 1: subjects skipped!\n")

            if STEP2_SL_METRICS:
                logger.info("### Step 2: computing SL-metrics started!\n")
                step_timer = Timer()

                subjects = utils.load_subjects(path_input=PATH, subject_pattern=SYN_PATTERN, debug=False)
                for subject in subjects:
                    logger.info("+++ Processing subject {}...".format(subject.get_ID()))
                
                    # check subject
                    graphs.check_subject(subject, check_ML=False)

                    # Compute SL graph metrics
                    subject = graphs.compute_sl_metrics(subject=subject, matrix_types=SL_MATRIX_TYPES, metric_list=METRIC_LIST, apply_threshold=True, logger=logger)

                    # Export subject
                    utils.save_subject(subject)

                logger.info("Process finished! ({} subjects processed)".format(len(subjects)))
                logger.info("### Step 2: finished with {}\n".format(step_timer))
            else:
                logger.warning("### Step 2: ML metrics skipped!\n")

            if STEP3_GNN:
                logger.info("### Step 3: trainning GNNs started!\n")
                step_timer = Timer()

                # create a list to store results
                global_results = []

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
                                for data_augmentation in [False, True]:
                                    # Set logger
                                    EXP_NAME = "VAE-FULL_"+ data_type_name + ("_DA_" if data_augmentation else "_noDA_") + MODEL + "-"+ str(DIM_H1) + "-" + str(DIM_H2) + "_lr" + str(LEARNING_RATE) + "_wd" + str(WEIGHT_DECAY)
                                    TIME_NAME = datetime.today().strftime("%Y%m%d-%H%M%S")
                                    LOGGING_NAME = TIME_NAME +"_"+ EXP_NAME
                                    logger.info("DA: {} | Model: {} | Dim_in: {} | Dim_out: {} | Dim_h1: {} | Dim_h2: {} | LR: {} | WD: {}".format(data_augmentation, MODEL, DIM_IN, DIM_OUT, DIM_H1, DIM_H2, LEARNING_RATE, WEIGHT_DECAY))
                                    
                                    # load data
                                    subjects = utils.load_subjects(path_input=PATH, subject_pattern=subject_pattern, debug=DEBUG)
                                    
                                    if NUM_CLASSES==2:
                                        target = [int(subject.get_mstype(type="binary")) for subject in subjects]
                                    else:
                                        target = [int(subject.get_mstype()) for subject in subjects]
                                    logger.info("MS Types: {}".format(sorted(np.unique(target))))
                                    # Matrix for prediction
                                    preds = np.zeros(len(subjects), dtype=int)

                                    # Data augmentation
                                    if data_augmentation:
                                        logger.info("*** USING Data Augmentation Strategy: VAE model")
                                        # Get all SYN subjects
                                        subjects_syn = utils.load_subjects(path_input=PATH, subject_pattern=SYN_PATTERN, debug=DEBUG)
                                        logger.info("   Loaded {} SYN subjects".format(len(subjects_syn)))
                                        
                                        # Filter subjects generated from the same cohort
                                        subjects_syn = [x for x in subjects_syn if x.get_cohort() == cohort]
                                        logger.info("   Filtering SYN subjects from {} cohort: remaining {} subjects".format(cohort, len(subjects_syn)))

                                        if NUM_CLASSES==2:
                                            target_syn = [int(x.get_mstype(type="binary")) for x in subjects_syn]
                                        else:
                                            target_syn = [int(x.get_mstype()) for x in subjects_syn]
                                        logger.info("   MS Types: {}".format(sorted(np.unique(target_syn))))
                                    else:
                                        logger.warning("*** NO Data Augmentation Strategy")
                                        subjects_syn = []
                                        target_syn = []

                                    # For each iteration
                                    glb_results = []

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

                                            # DA
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
                                            
                                            prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
                                            prop_test = np.where(y_test == 1)[0].shape[0] / y_test.shape[0]
                                            logger.info("Train set size     : {}".format(len(X_train)))
                                            logger.info("Test set size      : {}".format(len(X_test)))
                                            logger.info("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
                                            logger.info("Test set % of pwMS : {:.4f} ({})".format(prop_test, y_test.sum()))

                                            # list of Data structures (one for each subject)
                                            train_graphs = []
                                            for i in range(len(X_train)):
                                                g = gnn.array_to_graph(subject=X_train[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                                            num_classes=NUM_CLASSES, thr=THR[data_type_structure], logger=logger)
                                                train_graphs.append(g)
                                                
                                            # Test dataset
                                            test_graphs = []
                                            for i in range(len(X_test)):
                                                g = gnn.array_to_graph(subject=X_test[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                                            num_classes=NUM_CLASSES, thr=THR[data_type_structure], logger=logger)
                                                test_graphs.append(g)

                                            # create the model
                                            if MODEL == "GCN":
                                                model = GCN(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, logger=logger)
                                            elif MODEL == "GAT":
                                                model = GAT(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, heads=NUM_HEADS, dropout=DROPOUT, lr=LEARNING_RATE, 
                                                            wd=WEIGHT_DECAY, logger=logger)
                                            else:
                                                logger.error("Model not implemented!")
                                                raise NotImplementedError("Model not implemented!")
                                            
                                            model.fit(num_epocs=NUM_EPOCHS, train_graphs=train_graphs)
                                            y_pred = model.test(test_graphs=test_graphs, y_test=y_test)
                                            # Save predictions to compute final test scores
                                            preds[test_index] = y_pred

                                        if NUM_CLASSES == 2:
                                            # Compute metrics for binary classification
                                            auc_roc, _, _, _, _, _ = score.compute_auc_binary(target=target, preds=preds, logger=logger)
                                            #utils_score.plot_auc_roc_binary(target=target, preds=preds, name_out=LOGGING_NAME +"_"+ str(iteration+1))
                                        else:
                                            # Compute metrics
                                            score.compute_auc_multiclass(target=target, preds=preds, logger=logger)
                                            #utils_score.plot_auc_roc_multiclass(target=target, preds=preds, name_out=LOGGING_NAME +"_"+ str(iteration+1))
                                        
                                        # Add to global results
                                        glb_results.append(auc_roc)

                                    # Final AUC ROC score
                                    logger.info("AUC ROC scores: {}".format(glb_results))
                                    auc_roc_ave = np.mean(glb_results)
                                    auc_roc_std = np.std(glb_results)
                                    logger.info("*** Final AUC ROC score: {:.4f} +- {:.4f}".format(auc_roc_ave, auc_roc_std))

                                    # Store to results
                                    global_results.append({"NAME": EXP_NAME,
                                                        "DA": data_augmentation,
                                                        "Model": MODEL,
                                                        "Dim_in": DIM_IN,
                                                        "Dim_out": DIM_OUT,
                                                        "Dim_h1": DIM_H1,
                                                        "Dim_h2": DIM_H2,
                                                        "Learning_rate": LEARNING_RATE,
                                                        "Weight_decay": WEIGHT_DECAY,
                                                        "AUC_ROC_AVE": auc_roc_ave,
                                                        "AUC_ROC_STD": auc_roc_std})
                                    
                # Compute single score comparing with and without data augmentation
                diff = 0.0
                number = 0
                for i in range(0, len(global_results), 2):
                    assert global_results[i]["DA"] == False and global_results[i+1]["DA"] == True, "Results are not in the expected order (noDA, DA)"

                    auc_roc_ave_no_da = global_results[i]["AUC_ROC_AVE"]
                    auc_roc_std_no_da = global_results[i]["AUC_ROC_STD"]
                    auc_roc_ave_da = global_results[i+1]["AUC_ROC_AVE"]
                    auc_roc_std_da = global_results[i+1]["AUC_ROC_STD"]
                    diff += auc_roc_ave_da - auc_roc_ave_no_da
                    number += 1
                diff = diff/number if number > 0 else 0
                glb_acc_diff.append(diff)

                # Export to TXT file
                with open("results/"+ TIME_NAME +"_GNNs_"+ cohort +"_RS="+ str(RANDOM_SCALE) +".txt", "w") as f:
                    for line in global_results:
                        f.write("{} : AUC ROC = {:.4f} ± {:.4f} \n".format(line["NAME"], line["AUC_ROC_AVE"], line["AUC_ROC_STD"]))
                    f.write("\n*** AUC ROC difference (DA - noDA) = {:.4f}\n".format(diff))

                logger.info("*** AUC ROC difference (DA - noDA) = {:.4f}".format(diff))
                logger.info("### Step 3: finished with {}\n".format(step_timer))
            else:
                logger.warning("### Step 3: GNN skipped!\n")

            logger.info("### GLOBAL RESULTS: AUC ROC difference (DA - noDA) = {:.4f}\n".format(np.mean(glb_acc_diff)))
            logger.info("### PROCESS FINISHED! [Total running time = {}]\n".format(glb_timer))
            # Close logger
            handlers = logger.handlers[:]
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()