# The Next AI Revolution: A Tutorial Using VAEs to Generate High-Quality Synthetic Data
# https://towardsdatascience.com/the-next-ai-revolution-a-tutorial-using-vaes-to-generate-high-quality-synthetic-data/
#
# This script performs hyperparameter tuning for a Variational Autoencoder (VAE) using Optuna.
# It trains a VAE for each combination of cohort and matrix type, optimizing the latent dimension, hidden dimension, batch size, number of epochs, and learning rate to minimize the reconstruction mean squared
# error (MSE). The results are logged for each trial, and the best hyperparameters and performance are reported at the end of each training session.
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna
import logging
from datetime import datetime

from Subject import Subject
import utils.utils as utils
from utils.vae import *
from config import *


# PARAMETERS
# NOTE: The total number of VAE training runs will be N_TRAILS * len(COHORTS) * len(MATRIX_TYPES_ORIGINAL). 
# Adjust N_TRAILS accordingly to manage the total runtime.
# N. proves = N_TRIALS * len(COHORTS) * len(MATRIX_TYPES_ORIGINAL)
N_TRAILS = 100  # Number of Optuna trials
TIMEOUT = 36000  # Timeout in seconds
COHORTS = ["BCN", "NAP", "ALL"]
COHORTS_PATTERN = ["BCN-*.pkl", "NAP-*.pkl", "*.pkl"]


# Define the objective function and suggest hyperparameters values
def objective(trial):
    latent_dim = trial.suggest_int('latent_dim', 2, 32) # latent space dimension
    hidden_dim = trial.suggest_int('hidden_dim', 16, 64) # hidden layer dimension
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32]) # batch size
    num_epochs = trial.suggest_int('num_epochs', 100, 1000) # number of training epochs
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) # learning rate (log scale)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(np.array(X_train))), batch_size=batch_size, shuffle=True)

    # Initialize the VAE model with suggested hyperparameters
    model = BasicVAE(input_dim=INPUT_DIM, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    # Train the VAE model
    losses, mse = train_vae(model, train_loader, epochs=num_epochs, learning_rate=learning_rate)

    return mse[-1]  # Return the final MSE as the performance metric


if __name__ == "__main__":
    LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_Step-2-2-1_VAE_pretrain"
    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    for cohort, subject_pattern in zip(COHORTS, COHORTS_PATTERN):
        for matrix_type in MATRIX_TYPES_ORIGINAL:
            logging.info(f"=== Training VAE for cohort '{cohort}' ('{subject_pattern}') and {matrix_type} matrices ===")
            # Get dataset as flattened adjacency matrices
            X_train, y_train = get_dataset_as_flattened_adj_matrices(matrix_type=matrix_type, 
                                                                              subject_pattern=subject_pattern, 
                                                                              ms_type="binary",
                                                                              logger=logger)
            logging.info("Train set size: {}".format(len(X_train)))

            # Create a study object and optimize the objective function
            logging.info("Starting VAE training...")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=N_TRAILS) # timeout=TIMEOUT
            logging.info("Training completed! \n")

            # Print the best set of hyperparameters and performance
            logging.info(f"*** VAE :: cohort '{cohort}' ('{subject_pattern}') and {matrix_type}")
            logging.info('+++ Best hyperparameters:\n{}'.format(study.best_params))
            logging.info('+++ Best performance    :\n{}'.format(study.best_value))