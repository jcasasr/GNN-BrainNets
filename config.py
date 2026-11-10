import torch


##################################
### Paths and general settings ###
##################################
DEBUG = True
# Paths 
PATH_BCN = "/Users/jcasasr/Library/CloudStorage/OneDrive-ComputerVisionCenter/Datasets-MRI-Graph/CVC_MRI-Data/BCN_output"
PATH_NAP = "/Users/jcasasr/Library/CloudStorage/OneDrive-ComputerVisionCenter/Datasets-MRI-Graph/CVC_MRI-Data/NAP_output"
PATH = "./data"
# Matrix types
MATRIX_TYPES_ORIGINAL = ["FA", "GM", "RS"]
MATRIX_TYPES = ["FA", "GM", "RS", "ML", "FA-THR", "GM-THR", "RS-THR"] # Types of valid matrices
NUM_NODES = 76
# Thresholds for FA, GM, RS
THR = {'FA': 0.45, 'GM': 0.575, 'RS': 0.1,
       'FA-THR': 0.0, 'GM-THR': 0.0, 'RS-THR': 0.0}


def get_device(device:str=None):
    """
    Get the device to use for training (GPU if available, otherwise CPU).
    """
    if device:
        return device
    elif torch.cuda.is_available():
       return "cuda"
    elif torch.backends.mps.is_available():
       return "mps"
    else:
        return "cpu"