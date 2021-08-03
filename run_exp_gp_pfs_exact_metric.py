#
# DKAFT
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import shutil
import uuid
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import gpytorch
import torch
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from gp_layer import ExactGPModel
from plot_utils import residual_plot
from logging_conf import logger
from pytorchtools import EarlyStopping
from pml_pfs import run_metric_learning


def run(lr=3e-4, alpha=1e-3, n_hidden_sta=4, n_hidden_temp=32,
        model_name='lstm', n_embedding_temp=32, epoch=50, fold_idx=0,
        device=torch.device('cpu'), exp_name='dataset_xxx',
        run_name='model_xxx', seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    backbone, _ = run_metric_learning(batch_size=128,
                                      lr=lr, alpha=alpha,
                                      n_hidden_sta=n_hidden_sta,
                                      n_hidden_temp=n_hidden_temp,
                                      model_name=model_name,
                                      n_embedding_temp=n_embedding_temp,
                                      epoch=200, fold_idx=fold_idx,
                                      device=device,
                                      exp_name=exp_name,
                                      run_name=f'{model_name}_metrics',
                                      seed=seed
                                      )

    dp = Path(f'{str(Path.home())}/pfs/')
    fn = 'pfs_data.pt'

    data_fp = dp / fn
    split_df = pd.read_csv(dp / f'{fn[:-3]}_split.csv', index_col=0)

    data = torch.load(data_fp)

    X_padded, lengths, static_data, target = data

    X_padded = X_padded.to(device)
    lengths = lengths.to(device)
    static_data = static_data.to(device)
    target = target.to(device)
    
    x_dataset = TensorDataset(static_data, X_padded, lengths)
    y_dataset = TensorDataset(target)

    train_idx = split_df[split_df[f'fold_{fold_idx}'] == 1].index.to_list()
    valid_idx = split_df[split_df[f'fold_{fold_idx}'] == 2].index.to_list()
    test_idx = split_df[split_df[f'fold_{fold_idx}'] == 3].index.to_list()

    train_x = x_dataset[train_idx]
    train_y = y_dataset[train_idx][0]

    valid_x = x_dataset[valid_idx]
    valid_y = y_dataset[valid_idx][0]

    test_x = x_dataset[test_idx]
    test_y = y_dataset[test_idx][0]


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, backbone, likelihood)

    likelihood = likelihood.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(),
                                   'weight_decay': alpha,
                                   'lr': lr
                                   },
                                  {'params': model.covar_module.parameters()},
                                  {'params': model.mean_module.parameters()},
                                  {'params': likelihood.parameters()}
                                  ], lr=0.01)

    # step_scheduler = StepLR(optimizer, step_size=int(epoch/2), gamma=0.1)
    # scheduler = LRScheduler(step_scheduler)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    temp_name = f'temp_{uuid.uuid4()}'
    Path(temp_name).mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=25, verbose=True,
                                   path=f'./{temp_name}/checkpoint.pt')

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            'seed': seed,
            'num_epoch': epoch,
            'model': model_name,
            'weight_decay': alpha,
            'embedding_temp': n_embedding_temp,
            'hidden_dim_sta': n_hidden_sta,
            'hidden_dim_temp': n_hidden_temp,
            'fold_index': fold_idx,
            'file_data': fn,
            'pytorch version': torch.__version__,
            'cuda version': torch.version.cuda,
            'device name': torch.cuda.get_device_name(0)
        })

        iterator = tqdm(range(epoch))

        for i in iterator:

            model.train()
            likelihood.train()
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(*train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()

            model.eval()
            likelihood.eval()
            with torch.no_grad():
                valid_output = model(*valid_x)
                valid_y_pred = valid_output.mean.cpu().numpy()

            mse_valid = mean_squared_error(valid_y.cpu().numpy(), valid_y_pred)
            r2s_valid = r2_score(valid_y.cpu().numpy(), valid_y_pred)

            models = {'model': model,
                      'likelihood': likelihood}
            early_stopping(mse_valid, models)

            training_metrics = {'training mll': -loss.item(),
                                'validation mse': mse_valid,
                                'validation r2score': r2s_valid
                                }

            mlflow.log_metrics(training_metrics, step=i)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        # model.load_state_dict(torch.load(f'./{temp_name}/checkpoint.pt'))
        checkpoint = torch.load(f'./{temp_name}/checkpoint.pt')
        for k, m in models.items():
            m.load_state_dict(checkpoint[k])

        print('final test on the test set')
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            train_output = model(*train_x)
            train_y_pred = train_output.mean.cpu().numpy()
            valid_output = model(*valid_x)
            valid_y_pred = valid_output.mean.cpu().numpy()
            test_output = model(*test_x)
            test_y_pred = test_output.mean.cpu().numpy()

        mse_train = mean_squared_error(train_y.cpu().numpy(), train_y_pred)
        r2s_train = r2_score(train_y.cpu().numpy(), train_y_pred)
        mse_valid = mean_squared_error(valid_y.cpu().numpy(), valid_y_pred)
        r2s_valid = r2_score(valid_y.cpu().numpy(), valid_y_pred)
        mse_test = mean_squared_error(test_y.cpu().numpy(), test_y_pred)
        r2s_test = r2_score(test_y.cpu().numpy(), test_y_pred)
        test_metrics = {'training mse': mse_train,
                        'training r2score': r2s_train,
                        'validation mse': mse_valid,
                        'validation r2score': r2s_valid,
                        'test mse': mse_test,
                        'test r2score': r2s_test}
        mlflow.log_metrics(test_metrics, step=i)

        residual_plot(train_y.cpu().numpy(), train_y_pred,
                      valid_y.cpu().numpy(), valid_y_pred, dp=temp_name,
                      n_epoch=i, label='y_valid')
        residual_plot(train_y.cpu().numpy(), train_y_pred,
                      test_y.cpu().numpy(), test_y_pred, dp=temp_name,
                      n_epoch=i, label='y_test')

        mlflow.log_artifacts(f'./{temp_name}/')
        try:
            shutil.rmtree(temp_name)
        except FileNotFoundError:
            logger.warning('Temp drectory not found!')
            raise


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset parameter
    exp = 'Progression_free_survival'

    n_epoch = 400
    m_name = 'lstm'  # or 'gru'

    lrt = 1e-4  # this is from lr finder

    for f in range(5):
        run(lr=lrt, alpha=1e-7,
            n_hidden_sta=4,
            n_hidden_temp=128,
            model_name=m_name,
            n_embedding_temp=32,
            epoch=n_epoch,
            fold_idx=f, device=dc, exp_name=exp,
            run_name=f'{m_name}_exact_gp_metric',
            seed=sd)
