import os
import logging
from datetime import datetime

from Subject import Subject
import utils.utils as utils
import utils.graphs as graphs
from utils.timer import Timer
from config import *


##############
### PARAMS ###
##############
# Subject pattern to load subjects from PATH
# All: "*.pkl" | BCN: "BCN-*.pkl" | NAP: "NAP-*.pkl" | SYN: "SYN-*.pkl"
SUBJECT_PATTERN = "*.pkl"
# List of metrics to be computed
METRIC_LIST = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality', 'Clustering', 'EigenvectorCentrality', 'PageRank']
# List of single-layer matrix types to compute metrics for
SL_MATRIX_TYPES = ["FA", "GM", "RS", "FA-THR", "GM-THR", "RS-THR"]


if __name__ == "__main__":
    LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_Step-5-1_Compute_SL_Metrics"
    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

    logger.info(f"====================================================")
    logger.info(f"=== SUBJECT_PATTERN: {SUBJECT_PATTERN}")
    logger.info(f"====================================================\n")

    # timer
    step_timer = Timer()

    subjects = utils.load_subjects(path_input=PATH, subject_pattern=SUBJECT_PATTERN, debug=False)
    for subject in subjects:
        logger.info("+++ Processing subject {}...".format(subject.get_ID()))
    
        # check subject
        graphs.check_subject(subject=subject, logger=logger)

        # Compute SL graph metrics
        subject = graphs.compute_sl_metrics(subject=subject, matrix_types=SL_MATRIX_TYPES, 
                                            metric_list=METRIC_LIST, apply_threshold=False, logger=logger)

        # Export subject
        utils.save_subject(subject=subject)

    logger.info("=== Process finished! ({} subjects processed) ===\n".format(len(subjects)))
    logger.info(f"=== Total elapsed time: {step_timer} ===")
