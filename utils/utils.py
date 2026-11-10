import os
import pickle
from typing import List
import logging
from datetime import datetime
import fnmatch
import os
import numpy as np
import pandas as pd

from Subject import Subject
from config import *


def set_logging(logging_level=logging.DEBUG, logging_name=None):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # remove older loggers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    # Create a formatter to define the log format
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Create a file handler to write logs to a file
    if logging_name is None:
        logging_name = datetime.today().strftime("%Y%m%d-%H%M%S")

    file_handler = logging.FileHandler(filename=os.path.join("logs", logging_name + "_log.txt"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_object(obj, filename, debug=False):
    # Overwrites any existing file
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    if debug:
        print("   Subject {} exported!".format(obj.get_ID()))


def save_subject(subject, debug=False):
    out_filename = os.path.join(PATH, subject.get_ID() +".pkl")
    # Overwrites any existing file
    with open(out_filename, 'wb') as outp:
        pickle.dump(subject, outp, pickle.HIGHEST_PROTOCOL)
    if debug:
        print("   Subject {} exported!".format(subject.get_ID()))


def load_object(filename, debug=False):
    # Load the object from the file
    with open(filename, 'rb') as inp:
        objname = pickle.load(inp)
        if debug:
            print("Loading subject {}...".format(objname.get_ID()))

        return objname
    

def load_subjects(path_input:str="", subject_pattern:str="*.pkl", debug=False) -> List:
    files = sorted(os.listdir(path=path_input))
    subjects = []
    for file in files:
        if fnmatch.fnmatch(file, subject_pattern):
            subject = load_object(os.path.join(path_input, file), debug=debug)
            subjects.append(subject)
            if debug:
                print("Subject {} loaded!".format(subject.get_ID()))

    return subjects


def set_diagonal(matrix, value:float=0):
    """
    Set the diagonal of a matrix to a specific value.
    :param matrix: Matrix to set the diagonal
    :param value: Value to set the diagonal
    :return: Matrix with the diagonal set
    """
    np.fill_diagonal(matrix, value)
    return matrix