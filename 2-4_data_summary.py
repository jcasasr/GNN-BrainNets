import os
import pickle
import numpy as np
from typing import List
import logging
from datetime import datetime

from Subject import Subject
import utils.utils as utils
from config import *


# Parameters
COHORTS = ["BCN", "NAP", "SYN", "ALL"]
COHORTS_PATTERN = ["BCN-*.pkl", "NAP-*.pkl", "SYN-*.pkl", "*.pkl"]

if __name__ == "__main__":
    for cohort, pattern in zip(COHORTS, COHORTS_PATTERN):
        print(f"=== Processing cohort: {cohort} ===")
        # load data
        subjects = utils.load_subjects(path_input=PATH, subject_pattern=pattern, debug=False)
        subjects_hv = [subject for subject in subjects if subject.get_mstype(type="multiclass")==0]
        subjects_ms1 = [subject for subject in subjects if subject.get_mstype(type="multiclass")==1]
        subjects_ms2 = [subject for subject in subjects if subject.get_mstype(type="multiclass")==2]
        subjects_ms3 = [subject for subject in subjects if subject.get_mstype(type="multiclass")==3]

        print(f"Total subjects.   : {len(subjects)}")
        print(f"  - HV            : {len(subjects_hv)}")
        print(f"  - pwMS          : {len(subjects_ms1) + len(subjects_ms2) + len(subjects_ms3)}")
        print(f"    + type 1 RRMS : {len(subjects_ms1)}")
        print(f"    + type 2 SPMS : {len(subjects_ms2)}")
        print(f"    + type 3 PPMS : {len(subjects_ms3)}")
