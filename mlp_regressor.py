import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import copy

import pandas as pd

import seaborn as sns
import torch.nn as nn
import torch.optim as optim

import numpy as np
import wandb

import random
from scipy.stats import pearsonr


activations = {
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "gelu": nn.GELU(),
    "selu": nn.SELU()
}
optimizers = {
    "adam": optim.AdamW,
    "sgd": optim.SGD
}


class MLPRegressor(nn.Module):

    def __init__(self, device, input_size, output_size, hidden_layer_size, hidden_activation, last_activation, optimizer, learning_rate, max_iter):
        # Deterministic Run
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        super(MLPRegressor, self).__init__()
        self.max_iter = max_iter
        self.criterion = nn.MSELoss()
        self.hidden_activation = activations[hidden_activation]
        self.last_activation = activations[last_activation]
        self.device = device
        self.layers =  nn.ModuleList(
            [nn.Linear(input_size, 1, bias=False)] \
                + [nn.Linear(hidden_layer_size, output_size, bias=False)]
        )
        self.best_weights = []
        self.init_weights = []
        self._initialize_weights()
        self.optimizer = optimizers[optimizer](self.parameters(), lr=learning_rate)
        self.best_train_loss, self.best_test_pearson_corr = 1, -1
        self.initial_train, self.initial_test, self.final_train, self.final_test = None, None, None, None 

    def _initialize_weights(self):
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.init_weights.append(layer.weight)

        for layer in self.layers:
            self.best_weights.append(layer.weight)

    def set_to_best_weights(self):
        for i, weight in enumerate(self.best_weights):
            self.layers[i].weight = weight

    def forward(self, x): 
        if len(x.shape) == 2:
            x = self.layers[1](x)
            return self.last_activation(x) 
        else:
            # x = self.layer_norm(x)
            x = x.permute(0, 2, 1)
            x = self.layers[0](x)
            x = self.hidden_activation(x)
            x = x.permute(0, 2, 1)
            x = self.layers[1](x)
            x = x.squeeze(dim=-1)
            return self.last_activation(x) 

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
    
    def validate(self, X_test, y_test):
            preds = self.predict(X_test)
            test_loss = self.criterion(preds, y_test).item()

            result_df = pd.DataFrame(
                {
                    "preds": preds.squeeze(dim=-1).detach().cpu().numpy(), 
                    "target": y_test.squeeze(dim=-1).detach().cpu().numpy()
                }
            )
            test_spearman_corr = result_df.corr(method="spearman").iloc[0, 1]
            # test_pearson_corr = result_df.corr(method="pearson").iloc[0, 1]
            test_pearson_corr, test_pearson_p_value = pearsonr(result_df["preds"], result_df["target"])
            return result_df, test_loss, test_spearman_corr, test_pearson_corr, test_pearson_p_value

    def fit(self, X_train, y_train, X_test, y_test):     
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)  
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(dim=1).to(self.device)
        for epoch in range(self.max_iter):
            epoch_train_loss = 0.0
            n_examples = 0
            for batch in X_train:
                # Train
                self.optimizer.zero_grad()
                preds = self.forward(batch[0].to(torch.float32).to(self.device))
                train_loss = self.criterion(preds, batch[1].to(torch.float32).unsqueeze(dim=1).to(self.device))
                train_loss.backward()
                self.optimizer.step()
                epoch_train_loss += train_loss.item()
                n_examples += batch[0].shape[0]

            # Test 
            epoch_train_loss = epoch_train_loss / y_train.shape[0]
            result_df, test_loss, test_spearman_corr, test_pearson_corr, test_pearson_p_value = self.validate(X_test, y_test)
            
            if epoch == 0:
                self.initial_test = result_df
            if test_pearson_corr > self.best_test_pearson_corr:
                self.best_test_pearson_corr = test_pearson_corr
                self.best_weights = [copy.deepcopy(l.weight)for l in self.layers]
                self.final_test = result_df

            wandb.log({"test_loss": test_loss, "train_loss": epoch_train_loss, "test_spearman_corr": test_spearman_corr, "test_pearson_corr": test_pearson_corr})     
        wandb.finish() 
