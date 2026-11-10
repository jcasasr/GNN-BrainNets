import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_max_pool
from sklearn.metrics import roc_auc_score, f1_score


class ChebNet(torch.nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, dim_in:int, dim_out:int, dim_h1:int=32, k:int=3, lr:float=0.001, wd:float=5e-4, 
                 patience:int=10, delta:float=0.01, device:str='cpu', logger=None):
        """
        Constructor
        """ 
        super().__init__()

        ### Define the model
        # Layer 1: Conv
        self.conv1 = ChebConv(in_channels=dim_in, out_channels=dim_h1, K=k)
        self.conv2 = ChebConv(in_channels=dim_h1, out_channels=dim_h1, K=k)
        self.lin1 = torch.nn.Linear(dim_h1, dim_out)

        ### Parameters and settings
        self._learning_rate = lr
        self._weight_decay = wd
        self._batch_size = 32
        self._num_classes = dim_out
        # Set device
        self._device = device
        self.to(self._device)
        # Set logger
        self._logger = logger
        # Set early stopping parameters
        self._patience = patience
        self._delta = delta
        self._best_score = None
        self._early_stop = False
        self._counter = 0
        self._best_model_state = None

        # print info
        self._print_info()


    def forward(self, data):
        """
        Forward pass
        """
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        # Layer 1
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)

        # Dropout
        h = F.dropout(h, p=0.5, training=self.training)

        # Layer 2
        if self.conv2 is not None:
            h = self.conv2(h, edge_index, edge_weight)
            h = F.relu(h)

        # Max pooling
        h = global_max_pool(h, batch)
        
        # Classification layer
        h = self.lin1(h)

        return F.log_softmax(h, dim=1)
    
    
    def fit(self, num_epocs:int, train_graphs):
        """
        Train the model
        """
        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        # loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train for N epochs
        for epoch in range(num_epocs):
            self.train()
            train_loader = DataLoader(train_graphs, batch_size=self._batch_size, shuffle=True)
            y_pred, y_true = [], []

            loss_all = 0
            for batch in train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                output = self(batch)
                label = batch.y
                label = F.one_hot(label, num_classes=self._num_classes)
                label = label.type(torch.FloatTensor)
                label = label.to(self._device)
                loss = loss_fn(output, label)
                loss.backward()
                loss_all += batch.num_graphs * loss.item()
                optimizer.step()

                # Accuracy
                y_pred = y_pred + output.argmax(dim=1).tolist()
                y_true = y_true + batch.y.tolist()
            
            train_loss = loss_all / len(train_graphs)
            train_acc = self._accuracy(y_pred, y_true)
            self._logger.info("Epoch {:>3} : Train loss: {:.4f} | Acc: {:.2f} %".format(epoch + 1, train_loss, train_acc))

            if self.early_stopping(train_loss):
                self._logger.warning("Early stopping")
                self.load_best_model()
                break


    def early_stopping(self, val_loss) -> bool:
        """
        Early stopping
        """
        score = abs(val_loss)

        if self._best_score is None:
            # Initialize the best score and model state
            self._best_score = score
            self._best_model_state = self.state_dict()
        elif self._best_score + self._delta < score:
            # Increment the counter if the score has not improved
            self._counter += 1
            if self._counter >= self._patience:
                self._early_stop = True
                return True
        else:
            # Reset the counter and update the best score and model state if the score has improved
            self._best_score = score
            self._best_model_state = self.state_dict()
            self._counter = 0

        return False
    
    
    def load_best_model(self) -> bool:
        """
        Load the best model state
        """
        if self._best_model_state is not None:
            self.load_state_dict(self._best_model_state)
            self._logger.debug("Best model state loaded successfully.")
            return True
        else: 
            self._logger.error("No best model state found. Cannot load best model.")
            return False
    

    def test(self, test_graphs, y_test):
        """
        Test the model
        """
        test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
        
        for batch in test_loader:
            batch = batch.to(self._device)
            test_preds = F.softmax(self(batch), dim=1).detach().numpy()
        
        # Compute One Hot Encoding
        y_test_oh = F.one_hot(torch.from_numpy(y_test), num_classes=self._num_classes)

        # Compute AUC ROC
        auc_roc = roc_auc_score(y_test_oh, test_preds.round(), average='macro', multi_class='ovr')
        self._logger.info("Test AUC: {:.2f}".format(auc_roc))
        # Compute F1 Score
        f1 = f1_score(y_test_oh, test_preds.round(), average='weighted')
        self._logger.info("Test F1 Score: {:.2f}".format(f1))

        # Generate test_preds indices instead of probabilities (one hot encoding)
        test_preds_idx = np.argmax(test_preds, axis=1)

        return test_preds_idx
    
    def _accuracy(self, y_pred, y_true):
        """
        Calculate accuracy
        """
        return np.sum(np.array(y_pred) == np.array(y_true)) / len(y_true) * 100
    
    def _print_info(self):
        """
        Print model info
        """
        self._logger.info("++++++++++++++++++++++")
        self._logger.info("+++ Model: ChebNet +++")
        self._logger.info("++++++++++++++++++++++")
        self._logger.info("Device: {}".format(self._device))
        self._logger.info("Learning rate: {}".format(self._learning_rate))
        self._logger.info("Weight decay: {}".format(self._weight_decay))
        self._logger.info("Batch size: {}".format(self._batch_size))
        self._logger.info("Patience: {} and delta: {}".format(self._patience, self._delta))
        self._logger.info("Model architecture:")
        self._logger.info("Layer 1: GCNConv({}, {})".format(self.conv1.in_channels, self.conv1.out_channels))
        if self.conv2 is not None:
            self._logger.info("Layer 2: GCNConv({}, {})".format(self.conv2.in_channels, self.conv2.out_channels))
        self._logger.info("Layer 3: Linear({}, {})".format(self.lin1.in_features, self.lin1.out_features))


    def export_best_model(self, file_path:str):
        """
        Export the best model state to a file
        """
        if self._best_model_state is not None:
            torch.save(self._best_model_state, file_path)
            self._logger.info("Best model state exported successfully to {}".format(file_path))
        else:
            self._logger.error("No best model state found. Cannot export best model.")