import numpy as np
import pandas as pd
import os
import pickle
from typing import List

from Subject import Subject
from utils.utils import *
from config import *


def create_subjects(path_input:str="", path_output:str="", debug=False) -> List:
    """
    Create subjects from the input data.
    :param path_input: Path to the input data
    :param path_output: Path to the output data
    :param debug: Debug mode
    :return: List of subjects
    """
    # Load the data
    info = pd.read_csv(os.path.join(path_input, "ID_info.csv"))
    data_FA = np.load(os.path.join(path_input, "data", 'data_FA.npy'))
    data_GM = np.load(os.path.join(path_input, "data", 'data_GM.npy'))
    data_RS = np.load(os.path.join(path_input, "data", 'data_RS.npy'))

    # atts
    atts = ["gender", "edss", "dobirth", "doscan", "dostart", "age", "DD"]

    # List of subjects
    subjects = []

    for i in range(len(info)):
        print("Processing subject {}...".format(i))

        # Get the information
        ID = info.loc[i, "ID"]
        cohort = str(info.loc[i, "origin"])
        ID_old = str(info.loc[i, "ID_old"])
        ms_type = info.loc[i, "mstype"]

        # change the ID to "cohort-ID"
        ID = cohort + "-" + "{:04}".format(ID)

        # change mstype to multi-class labels
        # - Originally stored: -1: HV, 0: RRMS, 1: SPMS, 2: PPMS
        # - For binary classification, we will use 0: HV, 1: RRMS/SPMS/PPMS
        # - For multi-class classification, we will use 0: HV, 1: RRMS, 2: SPMS, 3: PPMS
        if ms_type == -1:
            ms_type = 0
        elif ms_type == 0:
            ms_type = 1
        elif ms_type == 1:
            ms_type = 2
        elif ms_type == 2:
            ms_type = 3
        else:
            raise Exception("Invalid MS type: {}".format(ms_type))

        # Create Subject object
        subject = Subject(ID, cohort, ID_old, ms_type)

        # Set attributes
        for att in atts:
            subject.set_attribute(att, info.loc[i, att])

        # Set diagonal to 0
        tmp_data_FA = set_diagonal(matrix=data_FA[i], value=0)
        tmp_data_GM = set_diagonal(matrix=data_GM[i], value=0)
        tmp_data_RS = set_diagonal(matrix=data_RS[i], value=0)
        
        # Set matrices
        subject.set_matrix("FA", np.abs(tmp_data_FA)) # don't threshold FA matrix: thr=THR["FA"]
        subject.set_matrix("GM", np.abs(tmp_data_GM)) # idem
        subject.set_matrix("RS", np.abs(tmp_data_RS)) # idem

        # Export subject
        save_subject(subject)

        # Append to the list
        subjects.append(subject)

        if debug:
            print(subject)

    return subjects


if __name__ == "__main__":
    # Load the subjects from BCN
    subjects = create_subjects(PATH_BCN, PATH, debug=True)
    print("Number of subjects: ", len(subjects))

    # Load the subjects from NAP
    subjects = create_subjects(PATH_NAP, PATH, debug=True)
    print("Number of subjects: ", len(subjects))