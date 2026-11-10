# The Next AI Revolution: A Tutorial Using VAEs to Generate High-Quality Synthetic Data
# https://towardsdatascience.com/the-next-ai-revolution-a-tutorial-using-vaes-to-generate-high-quality-synthetic-data/
#
# This script trains a Variational Autoencoder (VAE) for each combination of cohort and matrix type using the best hyperparameters found in the previous hyperparameter tuning step (2-2-1_VAE_pretrain.py). 
# The training process is logged, and the final reconstruction mean squared error (MSE) is visualized and saved for each trained model. 
# The trained models are also saved for future use.
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
from config import *

from utils.vae import *

# Parameters
COHORTS = ["BCN", "NAP", "ALL"]
COHORTS_PATTERN = ["BCN-*.pkl", "NAP-*.pkl", "*.pkl"]

if __name__ == "__main__":
    LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_Step-2-2-2_VAE_train"
    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)

    for cohort, subject_pattern in zip(COHORTS, COHORTS_PATTERN):
        for matrix_type in MATRIX_TYPES_ORIGINAL:
            logger.info(f"=== Training VAE for cohort '{cohort}' ('{subject_pattern}') and {matrix_type} matrices ===")
            # Get dataset as flattened adjacency matrices
            X_train, y_train = get_dataset_as_flattened_adj_matrices(matrix_type=matrix_type, 
                                                                            subject_pattern=subject_pattern, 
                                                                            ms_type="binary",
                                                                            logger=logger)
            logger.info("Train set size: {}".format(len(X_train)))

            # Create DataLoaders
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=PARAMS_VAE[cohort][matrix_type]["batch_size"], shuffle=True)
            
            vae = BasicVAE(input_dim=INPUT_DIM, 
                                 latent_dim=PARAMS_VAE[cohort][matrix_type]["latent_dim"], 
                                 hidden_dim=PARAMS_VAE[cohort][matrix_type]["hidden_dim"]).to(device)

            logger.info("Starting VAE training...")
            losses, mse = train_vae(vae, 
                                    train_loader, 
                                    epochs=PARAMS_VAE[cohort][matrix_type]["num_epochs"], 
                                    learning_rate=float(PARAMS_VAE[cohort][matrix_type]["learning_rate"]))
            logger.info("Training completed!")

            # Visualize results
            plt.figure(figsize=(12, 6))
            plt.plot(mse, label=f"VAE for {cohort} - {matrix_type}")
            plt.ylabel("Reconstruction MSE")
            plt.title("Training Reconstruction MSE")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'figs/VAE_training-mse_{cohort}_{matrix_type}.png')

            # Save the trained model
            model_name = f"VAE_{cohort}_{matrix_type}.pth"
            torch.save(vae.state_dict(), f'models/{model_name}')
            logger.info(f"Model saved as '{model_name}'")