#
# DKAFT
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners

import ignite.distributed as idist
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine, Events
from ignite.handlers import Checkpoint
from ignite.metrics import Metric
from ignite.utils import convert_tensor

if idist.has_xla_support:
    import torch_xla.core.xla_model as xm


class LinearModel(nn.Module):
    """Add a linear layer after the backbone."""
    def __init__(self, feature_extractor, num_features, output_dim=2):
        """

        Args:
            feature_extractor (nn.Module): The backbone of the model, used as a
                feature extractor.
            num_features (int): The number of features from the backbone.
            output_dim (int): The output dimension of the new model.
        """
        super(LinearModel, self).__init__()
        self.features = feature_extractor
        self.fc = nn.Linear(num_features, output_dim)

    def forward(self, x):
        features = self.features(x)
        out = self.fc(features)

        return out


class DKLModel(nn.Module):
    """Defines a DKL model with deep networks as a feature extractor and a
    GP based output layer for prediction."""
    def __init__(self, feature_extractor, gp_layer):
        """

        Args:
            feature_extractor (nn.Module): A sota feature extractor.
            gp_layer (nn.Module): A sota GP based output layer.
        """
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)

        return res


class SequenceFeature(nn.Module):
    """A feature extractor for static feature + sequential feature"""
    def __init__(self, n_feature_sta, n_feature_temp, n_hidden_sta,
                 n_hidden_temp, model='lstm', n_embedding_temp=None):
        """

        Args:
            n_feature_sta (int): The dimension of static features.
            n_feature_temp (int): The dimension of sequential features.
            n_hidden_sta (int): The dimension of the hidden static
                representations.
            n_hidden_temp (int): The dimension of the hidden sequential
                representations.
            model (str): The name of the sequential feature extractor,
                either lstm or gru.
            n_embedding_temp (int): The dimension of the temporal embeddings.
        """

        super(SequenceFeature, self).__init__()
        self.emb = nn.Linear(n_feature_sta, n_hidden_sta)

        if n_embedding_temp:
            self.temp_emb = nn.Linear(n_feature_temp, n_embedding_temp)
            n_feature_temp = n_embedding_temp
        else:
            self.temp_emb = None

        self.model = model
        if self.model == 'lstm':
            self.rnn = nn.LSTM(n_feature_temp, n_hidden_temp, batch_first=True)
        else:
            self.rnn = nn.GRU(n_feature_temp, n_hidden_temp, batch_first=True)

    def forward(self, x):
        sta, temp = x
        repr_sta = self.emb(sta)

        if self.temp_emb:
            temp = self.temp_emb(temp)

        if self.model == 'lstm':
            _, (h_n, _) = self.rnn(temp)
        else:
            _, h_n = self.rnn(temp)
        repr_temp = h_n.view(-1, h_n.size(2))
        repr_both = torch.cat((repr_sta, repr_temp), dim=1)

        return repr_both


class SequenceFeatureMCDropOut(nn.Module):
    """A feature extractor for static feature + sequential feature with
    dropout. """
    def __init__(self, n_feature_sta, n_feature_temp, n_hidden_sta,
                 n_hidden_temp, model='lstm', n_embedding_temp=None,
                 dropout_rate=0.2):
        """

        Args:
            n_feature_sta (int): The dimension of static features.
            n_feature_temp (int): The dimension of sequential features.
            n_hidden_sta (int): The dimension of the hidden static
                representations.
            n_hidden_temp (int): The dimension of the hidden sequential
                representations.
            model (str): The name of the sequential feature extractor,
                either lstm or gru.
            n_embedding_temp (int): The dimension of the temporal embeddings.
            dropout_rate (float): The value of the dropout rate.
        """

        super(SequenceFeatureMCDropOut, self).__init__()
        self.emb = nn.Linear(n_feature_sta, n_hidden_sta)

        if n_embedding_temp:
            self.temp_emb = nn.Linear(n_feature_temp, n_embedding_temp)
            n_feature_temp = n_embedding_temp
        else:
            self.temp_emb = None

        self.model = model
        if self.model == 'lstm':
            self.rnn = nn.LSTM(n_feature_temp, n_hidden_temp, batch_first=True)
        else:
            self.rnn = nn.GRU(n_feature_temp, n_hidden_temp, batch_first=True)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        sta, temp = x
        repr_sta = self.emb(sta)
        repr_sta = torch.nn.functional.dropout(repr_sta, p=self.dropout_rate)

        if self.temp_emb:
            temp = self.temp_emb(temp)
            temp = torch.nn.functional.dropout(temp, p=self.dropout_rate)

        if self.model == 'lstm':
            _, (h_n, _) = self.rnn(temp)
        else:
            _, h_n = self.rnn(temp)
        repr_temp = h_n.view(-1, h_n.size(2))
        repr_temp = torch.nn.functional.dropout(repr_temp, p=self.dropout_rate)
        repr_both = torch.cat((repr_sta, repr_temp), dim=1)

        return repr_both


class VSequenceFeature(nn.Module):
    """A feature extractor for static feature + sequential feature with
    variable-lenghts"""
    def __init__(self, n_feature_sta, n_feature_temp, n_hidden_sta,
                 n_hidden_temp, model='lstm', n_embedding_temp=None):
        """

        Args:
            n_feature_sta (int): The dimension of static features.
            n_feature_temp (int): The dimension of sequential features.
            n_hidden_sta (int): The dimension of the hidden static
                representations.
            n_hidden_temp (int): The dimension of the hidden sequential
                representations.
            model (str): The name of the sequential feature extractor,
                either lstm or gru.
            n_embedding_temp (int): The dimension of the temporal embeddings.
        """
        super(VSequenceFeature, self).__init__()
        self.emb = nn.Linear(n_feature_sta, n_hidden_sta)
        self.model = model

        if n_embedding_temp:
            self.temp_emb = nn.Linear(n_feature_temp, n_embedding_temp)
            n_feature_temp = n_embedding_temp
        else:
            self.temp_emb = None

        if self.model == 'lstm':
            self.rnn = nn.LSTM(n_feature_temp, n_hidden_temp, batch_first=True)
        else:
            self.rnn = nn.GRU(n_feature_temp, n_hidden_temp, batch_first=True)

    def forward(self, x):
        sta, temp, lengths = x
        repr_sta = torch.tanh(self.emb(sta))
        # repr_sta = self.emb(sta)

        if self.temp_emb:
            temp = torch.tanh(self.temp_emb(temp))
            # temp = self.temp_emb(temp)

        packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(temp,
                                                                   lengths.view(-1),
                                                                   batch_first=True,
                                                                   enforce_sorted=False)
        if self.model == 'lstm':
            _, (h_n, _) = self.rnn(packed_seq_batch)
        else:
            _, h_n = self.rnn(packed_seq_batch)

        repr_temp = h_n.view(-1, h_n.size(2))
        repr_both = torch.cat((repr_sta, repr_temp), dim=1)

        return repr_both


class VSequenceFeatureMCDropOut(nn.Module):
    """A feature extractora for static feature + sequential feature with
    variable-lenghts, dropout enabled during both training and evaluation. """
    def __init__(self, n_feature_sta, n_feature_temp, n_hidden_sta,
                 n_hidden_temp, model='lstm', n_embedding_temp=None,
                 dropout_rate=0.2):
        """

        Args:
            n_feature_sta (int): The dimension of static features.
            n_feature_temp (int): The dimension of sequential features.
            n_hidden_sta (int): The dimension of the hidden static
                representations.
            n_hidden_temp (int): The dimension of the hidden sequential
                representations.
            model (str): The name of the sequential feature extractor,
                either lstm or gru.
            n_embedding_temp (int): The dimension of the temporal embeddings.
            dropout_rate (float): The value of the dropout rate.
        """

        super(VSequenceFeatureMCDropOut, self).__init__()
        self.emb = nn.Linear(n_feature_sta, n_hidden_sta)
        self.model = model

        if n_embedding_temp:
            self.temp_emb = nn.Linear(n_feature_temp, n_embedding_temp)
            n_feature_temp = n_embedding_temp
        else:
            self.temp_emb = None

        if self.model == 'lstm':
            self.rnn = nn.LSTM(n_feature_temp, n_hidden_temp, batch_first=True)
        else:
            self.rnn = nn.GRU(n_feature_temp, n_hidden_temp, batch_first=True)

        self.dropout_rate = dropout_rate

    def forward(self, x):
        sta, temp, lengths = x
        repr_sta = torch.tanh(self.emb(sta))
        repr_sta = nn.functional.dropout(repr_sta, p=self.dropout_rate)

        if self.temp_emb:
            temp = torch.tanh(self.temp_emb(temp))
            temp = nn.functional.dropout(temp, p=self.dropout_rate)

        packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(temp,
                                                                   lengths.view(-1),
                                                                   batch_first=True,
                                                                   enforce_sorted=False)

        if self.model == 'lstm':
            _, (h_n, _) = self.rnn(packed_seq_batch)
        else:
            _, h_n = self.rnn(packed_seq_batch)

        repr_temp = h_n.view(-1, h_n.size(2))
        repr_temp = nn.functional.dropout(repr_temp, p=self.dropout_rate)
        repr_both = torch.cat((repr_sta, repr_temp), dim=1)

        return repr_both


class Mock(BaseEstimator):
    """Mock a BaseEstimator with defined prediction values
    """
    _estimator_type = "regressor"

    # Tell yellowbrick this is a regressor

    def __init__(self, y_pred_train, y_pred_test):
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test

    def predict(self, is_train=True):
        """X indicates whether prediction on train or not
        """
        if is_train:
            output = self.y_pred_train
        else:
            output = self.y_pred_test

        return output

    def score(self, X, y, sample_weight=None):

        y_pred = self.predict(X)

        return r2_score(y, y_pred, sample_weight=sample_weight)


class EpochOutputStore(object):
    """EpochOutputStore handler to save output prediction and target history
    after every epoch."""

    def __init__(self, output_transform=lambda x: x):
        """

        Args:
            output_transform (Callable): Transform the process_function's
            output_transform (Callable): Transform the process_function's
                output , e.g., lambda x: x[0].
        """
        self.predictions = None
        self.targets = None
        self.output_transform = output_transform

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, engine):
        y_pred, y = self.output_transform(engine.state.output)
        self.predictions.append(y_pred)
        self.targets.append(y)

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.update)

    def get_output(self, to_numpy=False):
        prediction_tensor = torch.cat(self.predictions, dim=0)
        target_tensor = torch.cat(self.targets, dim=0)

        if to_numpy:
            prediction_tensor = prediction_tensor.cpu().detach().numpy().flatten()
            target_tensor = target_tensor.cpu().detach().numpy().flatten()

        return prediction_tensor, target_tensor


class CheckPointAfter(Checkpoint):
    """Save the model after a defined epoch."""
    def __init__(self, start_epoch, *args, **kwargs):
        self.start_save_epoch = start_epoch
        print(f'start saving after {self.start_save_epoch}')
        super(CheckPointAfter, self).__init__(*args, **kwargs)

    def __call__(self, engine):
        global_step = self.global_step_transform(engine, engine.last_event_name)
        if global_step > self.start_save_epoch:
            super(CheckPointAfter, self).__call__(engine)
        else:
            print('skipping checkpoints...')


def get_initial_inducing_points_seq(feature_extractor, train_loader, device,
                                    num_inducing=5):
    """Generate initial inducing points using a backbone

    Args:
        feature_extractor (nn.Module): A backbone to generate features.
        train_loader (DataLoader): Dataloader of the
            training set.
        device (torch.device or str): Device to load the backbone and data.
        num_inducing (int): The multiple of batch size.
            The number of inducing points is (num_inducing x batch_size).

    Returns:
        torch.Tensor: The initial inducing points

    """
    feature_extractor.eval()
    inducing_points_list = []

    for i in range(num_inducing):
        with torch.no_grad():
            current_batch = next(iter(train_loader))
            sta, temp, _ = current_batch
            inducing_points = feature_extractor((sta.to(device),
                                                 temp.to(device))
                                                )
            inducing_points_list.append(inducing_points)

    initial_inducing_points = torch.cat(inducing_points_list, dim=0)

    # small_noise = torch.randn(initial_inducing_points.size()) * 0.1
    # small_noise = small_noise.to(device)
    # initial_inducing_points = initial_inducing_points + small_noise

    return initial_inducing_points


def get_initial_inducing_points_fasching(feature_extractor, train_loader,
                                         device, num_inducing=5):
    """Generate initial inducing points using a backbone

    Args:
        feature_extractor (nn.Module): A backbone to generate features.
        train_loader (DataLoader): Dataloader of the
            training set.
        device (torch.device or str): Device to load the backbone and data.
        num_inducing (int): The multiple of batch size.
            The number of inducing points is (num_inducing x batch_size).

    Returns:
        torch.Tensor: The initial inducing points

    """
    feature_extractor.eval()
    inducing_points_list = []

    for i in range(num_inducing):
        with torch.no_grad():
            current_batch = next(iter(train_loader))
            sta, temp, leng, _ = current_batch
            inducing_points = feature_extractor((sta.to(device),
                                                 temp.to(device),
                                                 leng.to(device))
                                                )
            inducing_points_list.append(inducing_points)

    initial_inducing_points = torch.cat(inducing_points_list, dim=0)

    return initial_inducing_points


################################################################################
"""Following are modified functions from ignite to facilitate the DKL 
training."""


def _prepare_batch(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def create_dkl_trainer(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mll: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False

    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        output = model(x)
        y_pred = output.mean.detach()
        loss = -mll(output, y)
        # loss = loss_fn(y_pred, y)
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def create_dkl_evaluator(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:

    metrics = metrics or {}

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model(x)
            y_pred = output.mean
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_metric_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    mining_function: miners.BaseMiner = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False

    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[
        Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        embeddings = model(x)
        indices_tuple = mining_function(embeddings, y)
        loss = loss_fn(embeddings, y, indices_tuple)
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        return output_transform(x, y, embeddings,
                                mining_function.num_triplets,
                                loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(
        _update)

    return trainer


if __name__ == '__main__':
    pass
