from enum import Enum
from typing import Dict, List
import numpy as np


class MSType(Enum):
    HV = 0
    RRMS = 1
    SPMS = 2
    PPMS = 3


class Subject():
    _ID: str
    _cohort: str
    _ID_old: str
    _mstype: int
    _attributes: Dict
    _matrices: Dict
    _metrics: Dict

    def __init__(self, ID:str, cohort:str, ID_old:str, mstype:int) -> None:
        self._ID = ID
        self._cohort = cohort
        self._ID_old = ID_old
        self._mstype = mstype
        self._attributes = {}
        self._matrices = {}
        self._metrics = {}

    ### Getters
    def get_ID(self) -> str:
        return self._ID

    def get_cohort(self) -> str:
        return self._cohort

    def get_ID_old(self) -> str:
        return self._ID_old

    def get_mstype(self, type:str="multiclass") -> int:
        '''
        Get the MS type of the subject.
        - type: "multiclass" or "binary". 
            If "multiclass", return the multi-class label (0: HV, 1: RRMS, 2: SPMS, 3: PPMS). 
            If "binary", return the binary label (0: HV, 1: RRMS/SPMS/PPMS).
        '''
        if type=="binary":
            return 0 if self._mstype==0 else 1
        else:
            return self._mstype

    ### Attributes
    def get_attribute(self, name:str="", raiseException:bool=True) -> str:
        if name in self._attributes:
            return self._attributes[name]
        else:
            if raiseException:
                raise ValueError("Attribute {} not found for subject {}".format(name, self._ID))
            return None

    def set_attribute(self, name:str, value:str) -> bool:
        self._attributes[name] = value
        return True

    ### Matrices
    def get_matrix(self, name:str="", raiseException:bool=True) -> np.ndarray:
        if name in self._matrices:
            return self._matrices[name]
        else:
            if raiseException:
                raise ValueError("Matrix {} not found for subject {}".format(name, self._ID))
            return None

    def set_matrix(self, name:str, value:np.ndarray, thr:float=0.0) -> bool:
        # set thresholded matrix if thr>0
        if thr>0:
            value[value < thr] = 0
        # set the matrix
        self._matrices[name] = value
        return True
    
    def get_all_matrices_keys(self) -> List[str]:
        return list(self._matrices.keys())

    ### Metrics
    def get_metric(self, name:str="", raiseException:bool=True) -> float:
        if name in self._metrics:
            return self._metrics[name]
        else:
            if raiseException:
                raise ValueError("Metric {} not found for subject {}".format(name, self._ID))
            return None

    def set_metric(self, name:str, value:float) -> bool:
        self._metrics[name] = value
        return True

    def __str__(self) -> str:
        return "Subject: {} --> Cohort: {} and MS type: {} [Old ID: {}]".format(self._ID, self._cohort, self._mstype, self._ID_old)