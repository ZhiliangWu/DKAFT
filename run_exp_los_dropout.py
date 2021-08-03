#
# DKAFT
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from functools import partial
import gc
from pathlib import Path
import shutil
import uuid
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import ignite
from ignite.contrib.handlers.mlflow_logger import MLflowLogger, \
    global_step_from_engine
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.handlers import DiskSaver, EarlyStopping
from ignite.metrics import MeanSquaredError

from data_utils import get_tensor_loaders_in_file, prepare_batch_tensor
from plot_utils import residual_plot
from logging_conf import logger
from model_utils import SequenceFeatureMCDropOut, LinearModel, EpochOutputStore, \
    CheckPointAfter


def run(batch_size=64, lr=3e-4, alpha=1e-3, n_hidden_sta=4, n_hidden_temp=32,
        model_name='lstm', n_embedding_temp=32, len_int=1, len_ts=24,
        epoch=50, fold_idx=0, device=torch.device('cpu'),
        exp_name='dataset_xxx', run_name='model_xxx', seed=42):
    """Run the experiment with a given setting and dropout.

    Args:
        batch_size (int): Batch size.
        lr (float): The value of the learning rate, possibly from lrfinder.
        alpha (float): The value of weight decay (a.k.a. regularization).
        n_hidden_sta (int): The dimension of the hidden static
                representations.
        n_hidden_temp (int): The dimension of the hidden sequential
            representations.
        model_name (str): The name of the backbone.
        n_embedding_temp (int): The dimension of the temporal embeddings.
        len_int (int): The length of the time interval (hours).
        len_ts (int):  The length of the number of time steps (hours).
        epoch (int): The number of training epochs.
        fold_idx (int): The index of the training/validation set.
        device (torch.device or str): The device to load the models.
        exp_name (str): The name of the experiments with a format of
            dataset_xxx, which defines the experiment name inside MLflow.
        run_name (str): The name of the run with a format of
            [model_name]_linear_regressor, which defines the run name inside
            MLflow.
        seed (int): The number of the random seed to ensure the reproducibility.

    Returns:
        None: The evolution of training loss and evaluation loss are saved to
            MLflow.

        """

    np.random.seed(seed)
    torch.manual_seed(seed)

    dp = Path(f'../mimic_setc')
    fn = f'imputed-normed-ep_{len_int}_{len_ts}_los.data'

    data_fp = dp / fn
    split_df = pd.read_csv(dp / f'{fn[:-5]}_split.csv', index_col=0)

    train_loader, valid_loader, test_loader, n_feature_sta, n_feature_temp = \
        get_tensor_loaders_in_file(data_fp, split_df,
                                   train_batch_size=batch_size,
                                   valid_batch_size=batch_size,
                                   n_fold=fold_idx)

    backbone = SequenceFeatureMCDropOut(n_feature_sta, n_feature_temp,
                                        n_hidden_sta=n_hidden_sta,
                                        n_hidden_temp=n_hidden_temp,
                                        model=model_name,
                                        n_embedding_temp=n_embedding_temp
                                        )

    n_features = n_hidden_sta + n_hidden_temp

    model = LinearModel(backbone, n_features, output_dim=1)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=alpha)

    # step_scheduler = StepLR(optimizer, step_size=int(epoch/2), gamma=0.1)
    # scheduler = LRScheduler(step_scheduler)

    def train_output_transform(x, y, y_pred, loss):
        return {'y': y, 'y_pred': y_pred, 'loss': loss.item()}

    trainer = create_supervised_trainer(model, optimizer,
                                        nn.MSELoss(),
                                        device=device,
                                        prepare_batch=prepare_batch_tensor,
                                        output_transform=train_output_transform
                                        )

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch loss': out['loss']})

    # evaluators
    val_metrics = {'mse': MeanSquaredError(),
                   'r2score': R2Score()
                   }

    for name, metric in val_metrics.items():
        metric.attach(trainer, name)

    evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                            device=device,
                                            prepare_batch=prepare_batch_tensor
                                            )
    pbar.attach(evaluator)

    eos_train = EpochOutputStore(output_transform=lambda out: (out['y_pred'],
                                                               out['y']))
    eos_valid = EpochOutputStore()
    eos_train.attach(trainer)
    eos_valid.attach(evaluator)

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):

        mlflow_logger = MLflowLogger()

        mlflow_logger.log_params({
            'seed': seed,
            'batch_size': batch_size,
            'num_epoch': epoch,
            'model': model_name,
            'weight_decay': alpha,
            'hidden_dim_sta': n_hidden_sta,
            'hidden_dim_temp': n_hidden_temp,
            'embedding_temp': n_embedding_temp,
            'fold_index': fold_idx,
            'file_data': fn,
            'pytorch version': torch.__version__,
            'ignite version': ignite.__version__,
            'cuda version': torch.version.cuda,
            'device name': torch.cuda.get_device_name(0)
        })

        # handlers for evaluator
        # note, this actually calls the evaluator
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(valid_loader)
            metrics = evaluator.state.metrics
            pbar.log_message(f"Validation Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared: {metrics['r2score']:.2f}"
                             )
            log_metrics = {f'validation {k}': v for k, v in metrics.items()}
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        temp_name = f'temp_{uuid.uuid4()}'

        def score_function(engine):
            return -engine.state.metrics['mse']

        to_save = {'model': model,
                   # 'optimizer': optimizer
                   }
        handler = CheckPointAfter(start_epoch=int(0.05 * epoch),
                                  to_save=to_save,
                                  save_handler=DiskSaver(f'./{temp_name}',
                                                         create_dir=True),
                                  n_saved=1,
                                  filename_prefix='best',
                                  score_function=score_function,
                                  score_name="val_mse",
                                  global_step_transform=global_step_from_engine(
                                      trainer))

        evaluator.add_event_handler(Events.COMPLETED, handler)

        es_handler = EarlyStopping(patience=20, score_function=score_function,
                                   trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, es_handler)

        # handlers for trainer
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            metrics = engine.state.metrics
            pbar.log_message(f"Training Set "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared: {metrics['r2score']:.2f}"
                             )

        def log_plots(engine, label='valid'):
            train_hist_y_p, train_hist_y = eos_train.get_output(to_numpy=True)
            val_hist_y_p, val_hist_y = eos_valid.get_output(to_numpy=True)

            residual_plot(train_hist_y, train_hist_y_p,
                          val_hist_y, val_hist_y_p, dp=temp_name,
                          n_epoch=engine.state.epoch, label=f'y_{label}')

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10),
                                  log_plots, label='valid')

        def final_evaluation(engine):
            to_load = to_save
            last_checkpoint_fp = f'./{temp_name}/{handler.last_checkpoint}'
            checkpoint = torch.load(last_checkpoint_fp, map_location=device)
            CheckPointAfter.load_objects(to_load=to_load, checkpoint=checkpoint)
            logger.info('The best model on validation is reloaded for '
                        'evaluation on the test set')
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            pbar.log_message(f"Testing Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared: {metrics['r2score']:.2f}"
                             )

            log_metrics = {f'test {k}': v for k, v in metrics.items()}
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        trainer.add_event_handler(Events.COMPLETED, final_evaluation)
        trainer.add_event_handler(Events.COMPLETED, log_plots, label='test')

        @trainer.on(Events.COMPLETED)
        def save_model_to_mlflow(engine):
            mlflow_logger.log_artifacts(f'./{temp_name}/')
            try:
                shutil.rmtree(temp_name)
            except FileNotFoundError:
                logger.warning('Temp drectory not found!')
                raise

        # log training loss at each iteration
        mlflow_logger.attach_output_handler(trainer,
                                            event_name=Events.ITERATION_COMPLETED,
                                            tag='training',
                                            output_transform=lambda out: {
                                                'batch_loss': out['loss']}
                                            )

        # setup `global_step_transform=global_step_from_engine(trainer)` to
        # take the epoch of the `trainer` instead of `train_evaluator`.
        mlflow_logger.attach_output_handler(trainer,
                                            event_name=Events.EPOCH_COMPLETED,
                                            tag='training',
                                            metric_names=['mse',
                                                          'r2score']
                                            )

        # Attach the logger to the trainer to log optimizer's parameters,
        # e.g. learning rate at each iteration
        mlflow_logger.attach_opt_params_handler(trainer,
                                                event_name=Events.ITERATION_STARTED,
                                                optimizer=optimizer,
                                                param_name='lr'
                                                )

        _ = trainer.run(train_loader, max_epochs=epoch)


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset parameter
    exp = 'mimic_los_SetC'
    interval = 1
    length_time = 48

    # fold = 0
    bs = 256
    n_epoch = 300

    lrt = 3e-4  # this is from lr finder

    # model building/training parameters
    # found through grid search
    n_hidden_s = 4
    n_hidden_t = 64
    n_embedding_temp = 64
    a = 1e-2
    m_name = 'gru'  # or 'gru'

    r_name = f'{m_name}_linear_regressor_dropout'

    for f in range(5):
        run(batch_size=bs, lr=lrt, alpha=a, n_hidden_sta=n_hidden_s,
            n_hidden_temp=n_hidden_t, model_name=m_name,
            n_embedding_temp=n_embedding_temp, len_int=interval,
            len_ts=length_time, epoch=n_epoch, fold_idx=f, device=dc,
            exp_name=exp, run_name=r_name, seed=sd)

        gc.collect()
