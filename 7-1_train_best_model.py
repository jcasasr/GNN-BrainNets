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
NUM_CLASSES = 2
# Experiment settings
N_ITERATIONS = 4
NUM_EPOCHS = 400
N_SPLITS = 4
# Early stopping parameters
PATIENCE = 20
DELTA = 0.01
# Real dataset (no use "*.pkl" to avoid loading synthetic subjects for testing)
SUBJECT_COHORT = ["ALL"]
SUBJECT_PATTERNS = ["BCN-*.pkl", "NAP-*.pkl"] # (do not use "*.pkl" to avoid loading synthetic subjects for testing)
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
# - Multi-modal: ("MM", "FA", ["FA", "GM", "RS"])
# - Multi-modal Thresholded: ("MM-THR", "FA-THR", ["FA-THR", "GM-THR", "RS-THR"])
TEST_METHODS = [("MM", "FA", ["FA", "GM", "RS"]),
                ("MM-THR", "FA-THR", ["FA-THR", "GM-THR", "RS-THR"])]
# List of metrics to be used as embeddings for GNNs
METRIC_LIST = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality', 
               'Clustering', 'EigenvectorCentrality', 'PageRank']

##################
### GNN params ###
##################
# Tuple: ("model_name", [dim_h1, dim_h2])
MODEL_ARCHITECTURES = [("FPGCN", [0, 0])]
LEARNING_RATES = [0.0001]
WEIGHT_DECAYS = [5e-4]
# GAT params
NUM_HEADS = 4
DROPOUT = 0.6
# Data augmentation
DATA_AUGMENTATION = False
DA_PATTERN = "SYN-*.pkl"


if __name__ == "__main__":
    # Timer
    glb_timer = Timer()
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
                    # Local timer
                    lcl_timer = Timer()
                    # Set logger
                    EXP_NAME = "Step-7-1_"+ data_type_name +"_"+ MODEL +"-"+ str(DIM_H1) +"-"+ str(DIM_H2) +"_lr"+ str(LEARNING_RATE) +"_wd"+ str(WEIGHT_DECAY) +("_DA" if DATA_AUGMENTATION else "_noDA")
                    TIME_NAME = datetime.today().strftime("%Y%m%d-%H%M%S")
                    LOGGING_NAME = TIME_NAME +"_"+ EXP_NAME
                    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

                    logger.info(f"====================================================")
                    logger.info(f"=== EXPERIMENT  : {EXP_NAME}")
                    logger.info(f"=== COHORT      : {SUBJECT_COHORT} ('{SUBJECT_PATTERNS}')")
                    logger.info(f"=== DATA_AUGMENT: {DATA_AUGMENTATION} (Pattern: '{DA_PATTERN}')")
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
                    
                    # Target for real subjects
                    target = [int(subject.get_mstype(type="binary")) for subject in subjects]
                    logger.info("MS Types: {}".format(sorted(np.unique(target))))

                    # All subjects for training
                    X_train = subjects
                    y_train = np.array(target)

                    # If data augmentation is enabled, load synthetic subjects
                    if DATA_AUGMENTATION:
                        logger.info("*** USING Data Augmentation Strategy: VAE model")
                        
                        # Get all SYN subjects
                        subjects_syn = utils.load_subjects(path_input=PATH, subject_pattern=DA_PATTERN, debug=DEBUG)
                        logger.info("   Loaded {} SYN subjects".format(len(subjects_syn)))

                        # Target for synthetic subjects
                        target_syn = [int(x.get_mstype(type="binary")) for x in subjects_syn]
                        logger.info("   MS Types: {}".format(sorted(np.unique(target_syn))))

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
                    else:
                        logger.info("*** NO Data Augmentation Strategy")
                        subjects_syn = []
                        target_syn = []

                    prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
                    logger.info("Train set size     : {}".format(len(X_train)))
                    logger.info("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
                    
                    # list of Data structures (one for each subject)
                    train_graphs = []
                    for i in range(len(X_train)):
                        g = gnn.array_to_graph(subject=X_train[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                    num_classes=NUM_CLASSES, thr=THR[data_type_structure], metric_list=METRIC_LIST, logger=logger)
                        train_graphs.append(g)
                    gnn.print_graphs_info(graphs=train_graphs, name="Train", thr=THR[data_type_structure], logger=logger)
                        
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
                        model = FPGCN(dim_in=DIM_IN, dim_out=DIM_OUT, lr=LEARNING_RATE, wd=WEIGHT_DECAY, 
                                    patience=PATIENCE, delta=DELTA, device=device, logger=logger)
                    else:
                        logger.error("Model not implemented!")
                        raise NotImplementedError("Model not implemented!")

                    # Train the model with the whole dataset
                    model.fit(num_epocs=NUM_EPOCHS, train_graphs=train_graphs)

                    # Save model
                    model.export_best_model("./models/"+ EXP_NAME +".pt")
                    logger.info("### MODEL SAVED: ./models/"+ EXP_NAME +".pt")

    logger.info("### PROCESS FINISHED! [Total running time = {}]\n".format(glb_timer))
