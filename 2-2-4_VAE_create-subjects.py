# The Next AI Revolution: A Tutorial Using VAEs to Generate High-Quality Synthetic Data
# https://towardsdatascience.com/the-next-ai-revolution-a-tutorial-using-vaes-to-generate-high-quality-synthetic-data/
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

from Subject import Subject
import utils.utils as utils
from utils.timer import Timer
from config import *

from utils.vae import *


# Parameters
NUM_SAMPLES = {0: 50, 1: 50, 2: 0, 3: 0}  # Number of synthetic samples to generate / ms_type
RANDOM_SCALE = 0.1  # Scale of the random noise added to the mean latent vector
COHORTS = ["ALL"] # "BCN", "NAP", "ALL"
COHORTS_PATTERN = ["*.pkl"] # "BCN-*.pkl", "NAP-*.pkl", "*.pkl"
COHORT_NEW = "SYN"


if __name__ == "__main__":
    LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_Step-2-2-4_VAE_create-subjects"
    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

    # Subject index
    subj_idx = 0
    # timer
    step_timer = Timer()

    # For each cohort
    for cohort, subject_pattern in zip(COHORTS, COHORTS_PATTERN):
        logger.info(f"====================================================")
        logger.info(f"=== COHORT      : {cohort} ('{subject_pattern}')")
        logger.info(f"=== NUM_SAMPLES : {NUM_SAMPLES}")
        logger.info(f"=== RANDOM_SCALE: {RANDOM_SCALE}")
        logger.info(f"=== DEVICE      : {device}")
        logger.info(f"====================================================\n")
        
        # For each MS type
        for ms_type in [0, 1, 2, 3]:
            num_samples = NUM_SAMPLES[ms_type]
            logger.info(f"Creating {num_samples} subjects for {cohort} ('{subject_pattern}') and ms_type={ms_type} by using VAE")
            
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

        logger.info(f"=== Finished creating subjects for {cohort} ('{subject_pattern}') ===")
        logger.info(f"=== Total subjects created: {subj_idx} ===\n")

    logger.info(f"=== Total elapsed time: {step_timer} ===")
