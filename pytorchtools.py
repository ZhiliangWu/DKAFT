#
# DKAFT
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

# modified from https://github.com/Bjarten/early-stopping-pytorch
# change to save checkpoint of a dict of models instead of a single model

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
            verbose (bool): If True, prints a message for each validation loss
                improvement.
            delta (float): Minimum change in the monitored quantity to qualify
                as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): Trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, models):
        """note, models are a dict of models"""

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models)
            self.counter = 0

    def save_checkpoint(self, val_loss, models):
        """Saves models when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        model_state_dict = {k: v.state_dict() for k, v in models.items()}
        torch.save(model_state_dict, self.path)
        self.val_loss_min = val_loss
