import os
import logging
from datetime import datetime

from Subject import Subject
import utils.utils as utils
import utils.graphs as graphs
from utils.timer import Timer
from config import *


###############
### PARAMS ####
###############
SUBJECT_PATTERN = "*.pkl"


if __name__ == "__main__":
    LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_Step-4_Data-Models-Creation"
    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

    logger.info(f"====================================================")
    logger.info(f"=== SUBJECT_PATTERN: {SUBJECT_PATTERN}")
    logger.info(f"====================================================\n")

    # timer
    step_timer = Timer()

    subjects = utils.load_subjects(path_input=PATH, subject_pattern=SUBJECT_PATTERN, debug=False)
    for subject in subjects:
        logger.info("+++ Processing subject {}...".format(subject.get_ID()))

        # Create multilayer
        ML = graphs.create_multilayer(subject.get_matrix("FA"), subject.get_matrix("GM"), subject.get_matrix("RS"))
        if ML is not None:
            subject.set_matrix(name="ML", value=ML)
            logger.info("\tMultilayer ML created for subject {}".format(subject.get_ID()))
        else:
            logger.error("\tError creating multilayer ML for subject {}".format(subject.get_ID()))
            raise ValueError("Error creating multilayer ML for subject {}".format(subject.get_ID()))

        # Define thresholded multilayer
        ML_THR = graphs.create_multilayer(subject.get_matrix("FA-THR"), subject.get_matrix("GM-THR"), subject.get_matrix("RS-THR"))
        if ML_THR is not None:
            subject.set_matrix(name="ML-THR", value=ML_THR)
            logger.info("\tThresholded multilayer ML-THR created for subject {}".format(subject.get_ID()))
        else:
            logger.error("\tError creating thresholded multilayer ML-THR for subject {}".format(subject.get_ID()))
            raise ValueError("Error creating thresholded multilayer ML-THR for subject {}".format(subject.get_ID()))
        # check subject
        graphs.check_subject(subject=subject, logger=logger)

        # Export subject
        utils.save_subject(subject=subject)

    logger.info("=== Process finished! ({} subjects processed) ===".format(len(subjects)))
    logger.info("=== Total time: {} ===\n".format(step_timer.get_time_str()))
