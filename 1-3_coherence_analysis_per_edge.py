from Subject import Subject
from config import *
import utils.utils as utils
import os
import numpy as np
from scipy.stats import chisquare, ttest_ind
import logging
from datetime import datetime


def t_test_per_edge(dataset1, dataset2, label=""):
    """
    Perform a t-test between two datasets, to see if they come from the same distribution.

    Parameters:
    dataset1 (list or array-like): First dataset.
    dataset2 (list or array-like): Second dataset.

    Returns:
    tuple: t-statistic and p-value of the t-test.
    """
    # Set logger
    EXP_NAME = "COHORT_ANALYSIS_" + label
    TIME_NAME = datetime.today().strftime("%Y%m%d-%H%M%S")
    LOGGING_NAME = TIME_NAME +"_"+ EXP_NAME
    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

    logger.info("+++ Matrix: {}".format(label))
    logger.info("+++ Performing t-test per edge between Dataset 1 and 2 cohorts")
    logger.info("+++ Dataset 1: {} subjects".format(len(dataset1)))
    logger.info("+++ Dataset 2: {} subjects".format(len(dataset2)))
    logger.info("")
    total = 0
    total_neg = 0

    for i in range(76):
        for j in range(i + 1, 76):
            logger.info(f"Edge {i}-{j}")

            # Get values for BCN and NAP cohorts
            bcn_values = np.array([x[i, j] for x in dataset1 if x is not None])
            nap_values = np.array([x[i, j] for x in dataset2 if x is not None])

            # Calculate means and standard deviations
            bcn_mean = np.mean(bcn_values) 
            nap_mean = np.mean(nap_values) 
            bcn_std = np.std(bcn_values) 
            nap_std = np.std(nap_values) 

            logger.info(f"BCN: mean={bcn_mean:.4f}, std={bcn_std:.4f} <-> NAP: mean={nap_mean:.4f}, std={nap_std:.4f}")
            
            bcn_values = bcn_values.reshape((len(bcn_values),1))
            nap_values = nap_values.reshape((len(nap_values),1))

            # Perform chi-square test
            result = ttest_ind(bcn_values, nap_values, axis=0)

            logger.info(f"Chi-Square Statistic: {result.statistic}")
            logger.info(f"P-value: {result.pvalue}")

            if result.pvalue < 0.05:
                logger.info("H0 is true. The distributions are likely DIFFERENT.")
                total += 1
            else:
                logger.info("Cannot discard H0. The distributions are likely SIMILAR.")
                total_neg += 1

    logger.info("\n")
    logger.info("+++ Total significant differences found: {} of {} ({:.4f} %)".format(total, total + total_neg, total / (total + total_neg) * 100))
    logger.info("+++ Total negative tests found         : {} of {} ({:.4f} %)".format(total_neg, total + total_neg, total_neg / (total + total_neg) * 100))


# Get matrix list
matrix_list = ["FA", "GM", "RS"]

# load data
subjects_BCN = utils.load_subjects(path_input=PATH, subject_pattern="BCN-*.pkl", debug=False)
subjects_NAP = utils.load_subjects(path_input=PATH, subject_pattern="NAP-*.pkl", debug=False)

# For each matrix type
for matrix in matrix_list:
    # All subjects for each cohort
    values_BCN = [x.get_matrix(name=matrix) for x in subjects_BCN]
    values_NAP = [x.get_matrix(name=matrix) for x in subjects_NAP]

    t_test_per_edge(values_BCN, values_NAP, label=matrix +"_All")

    # Compare HV of the two cohorts
    values_BCN = [x.get_matrix(name=matrix) for x in subjects_BCN if x.get_mstype(type="binary") == 0]
    values_NAP = [x.get_matrix(name=matrix) for x in subjects_NAP if x.get_mstype(type="binary") == 0]

    t_test_per_edge(values_BCN, values_NAP, label=matrix +"_HV")

    # Compare PwMS of the two cohorts
    values_BCN = [x.get_matrix(name=matrix) for x in subjects_BCN if x.get_mstype(type="binary") > 0]
    values_NAP = [x.get_matrix(name=matrix) for x in subjects_NAP if x.get_mstype(type="binary") > 0]

    t_test_per_edge(values_BCN, values_NAP, label=matrix +"_PwMS")

    
