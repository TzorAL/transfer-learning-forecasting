#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)

# Let's import some basic libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("webagg")

import torch
import torch.nn as nn
import pytorch_lightning as pl

import logging
import pickle
import click
import mlflow
from mlflow.models.signature import infer_signature
import os

import torchensemble 
from torchensemble.utils import io
import re
import tempfile

# Custom made classes and functions used for reading/processing data at input/output of model
from model_utils import ClickParams, Regression
from model_utils import read_csvs, train_test_valid_split, feature_target_split, \
                        cross_plot_actual_pred, calculate_metrics
# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

test_X,test_Y = None, None

def findNewRows(N,K):
    """
    Function to find the largest number
    smaller than or equal to N
    that is divisible by k
    """
    rem = N % K
    if (rem == 0):
        return N
    else:
        return N - rem

def ensemble(load=True, save_dir='./ensemble_models/', ensemble_filename='VotingRegressor_Regression_10_ckpt.pth'):
    """
    This function performs the ensembling method
    1. Reads models params from user arguments
    2. Constructs the custom pl model
    3. create ensemble model using torchensemble library
    4. evaluates model using MSE loss
    5. creates 1-D list for data targets and its relative predictions

    Parameters:
        load:   boolean for checking if ensemble model params should be taken from
                ensemble_filename or not
        save_dir:   directory path (relative to current path) for storing fully trained
                    ensemble model
        ensemble_filename: file path (relative to current path) for loading fully trained
                    ensemble model
    Returns:
        actuals: 1-D list containing target labels
        predictions:  1-D list containing predictions for said labels
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.set_num_threads(click_params.num_workers) ######################################
    pl.seed_everything(click_params.seed, workers=True)  

    best_params = {}; best_params.update(vars(click_params)); del best_params['n_estimators']
    
    # layer_sizes must be iterable for creating model layers
    if(not isinstance(best_params['layer_sizes'],list)):
        best_params['layer_sizes'] = [best_params['layer_sizes']] 
    
    # double asterisk (dictionary unpacking)
    regression_model = Regression(**best_params)
    regression_model.to(device)

    # At the end of constructor. set feature/target of model
    # create datasets used by dataloaders
        # 'subset'_X: dataset containing features of subset (train/test/validation) dataframe
        # 'subset'_Y: dataset containing targets of subset (train/test/validation) dataframe
    global test_X,test_Y # used by MASE metric 
    train_X, train_Y = feature_target_split(train_data,click_params.l_window,click_params.f_horizon)  
    test_X, test_Y = feature_target_split(test_data,click_params.l_window,click_params.f_horizon)
    validation_X, validation_Y = feature_target_split(val_data,click_params.l_window,click_params.f_horizon)      

    train_loader = regression_model.train_dataloader(train_X,train_Y)
    test_loader = regression_model.test_dataloader(test_X,test_Y)
    val_loader = regression_model.val_dataloader(validation_X,validation_Y)

    ensemble_model = None
    if(load==True and os.path.exists(save_dir)):
        # parse filename (e.g 'VotingRegressor_Regression_10_ckpt.pth') for:
        #   numeric value of n_estimators (10)
        #   regressor used (VotingRegressor)
        n_estimators = re.findall("\d+", ensemble_filename)[0]
        regressor = ensemble_filename.split('_')[0]
        
        # Define the ensemble
        ensemble_model = getattr(torchensemble,regressor)(
            estimator=regression_model,
            n_estimators=n_estimators, 
            cuda=True if torch.cuda.is_available() else False
        )
        io.load(ensemble_model, save_dir)  # reload
    else:
        # Define the ensemble
        ensemble_model = torchensemble.VotingRegressor(
            estimator=regression_model,
            n_estimators=click_params.n_estimators,
            cuda=True if torch.cuda.is_available() else False 
        )
        ensemble_model.to(device)
        
        # Set the criterion
        criterion = nn.MSELoss(); criterion.to(device)
        # criterion = MeanAbsolutePercentageError(); criterion.to(device)
        ensemble_model.set_criterion(criterion)

        # Set the optimizer
        ensemble_model.set_optimizer(optimizer_name=click_params.optimizer_name,
                                    lr=click_params.l_rate)

        ensemble_model.fit(epochs=click_params.max_epochs,
                        train_loader=train_loader,
                        test_loader=val_loader,
                        save_model=True,
                        save_dir=f'{save_dir}') #save to temp_dir (mlflow) or to local path

        # Determine incremented filename
        i = 0
        while os.path.exists(f"estimarors_{i}.pkl"):
            i += 1

        estimators_file = open(f"estimators_{i}.pkl", "w")

        for module in range(len(ensemble_model.estimators_)):
            print(ensemble_model.estimators_[module].named_parameters)
            # estimators_file.write(ensemble_model.estimators_[module].named_parameters)

    mse_loss = ensemble_model.evaluate(test_loader)
    print(f'Testing mean squared error of the fitted ensemble: {mse_loss}')

    actuals_X = []
    actuals_Y = []
    for x,y in test_loader: 
        actuals_X.append(x)
        actuals_Y.append(y)

    # actuals: tensor of tensor-batches
    # actuals[0]: first tensor-batch
    # actuals[0][0]: first feature/target of first batch

    # torch.vstack: convert list of tensors to (rank 2) tensor
    # .tolist(): convert (rank 2) tensor to list of lists
    # final outcome: list of floats
    actuals_X = [item for sublist in torch.vstack(actuals_X).tolist() for item in sublist]
    actuals_Y = [item for sublist in torch.vstack(actuals_Y).tolist() for item in sublist]

    # find max len of list (rows of later np.array) that is 
    # divisible by lookback_window to be fed as batches to predict
    new_len = findNewRows(len(actuals_X),click_params.l_window)
    actuals_X = actuals_X[:new_len]
    # print(f"new_len: {new_len}")

    # reshape 1-D matrix (created by list train_X) for it to be multipled by input layer matrix
    # matrix shape: [any, lookback_window]
    preds = ensemble_model.predict(np.array(actuals_X).reshape(-1, click_params.l_window))

    # store BaggingRegressor in mlflow as artifact "ensemble_model" with input singature
    signature = infer_signature(train_X.head(1), pd.DataFrame(preds)) 
    mlflow.pytorch.log_model(ensemble_model, "ensemble_model", signature=signature)

    # convert preds to 1-D list
    # keep in actuals_Y, the targets of features used by predict()
    preds = [item for sublist in preds.tolist() for item in sublist]
    actuals_Y = actuals_Y[:len(preds)]

    return preds, actuals_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--dir_in", type=str, default='../preprocessed_data/', help="File containing csv files used by the model")
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option("--seed", type=str, default="42", help='seed used to set random state to the model')
@click.option('--train_years', type=str, default='2015,2016,2017,2018,2019', help='list of years to use for training set')
@click.option('--val_years', type=str, default='2020', help='list of years to use for validation set')
@click.option('--test_years', type=str, default='2021', help='list of years to use for testing set')
@click.option("--max_epochs", type=str, default="2", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1", help='range of number of layers used by the model')
@click.option("--layer_sizes", type=str, default="100", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="240", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24", help='range of forecast horizon (output layer size) used by the model')
@click.option("--l_rate", type=str, default="1e-4", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="1024", help='possible batch sizes used by the model') #16,32,
@click.option("--transfer_mode", type=str, default="0", help='indicator to use transfer learning techniques')
@click.option("--n_estimators", type=str, default="3", help='number of estimators (models) used in ensembling')
@click.option("--num_workers", type=str, default="4", help='accelerator (cpu/gpu) processesors and threads used') 
@click.option('--time_steps', type=str, default='168', help='naive model time lags')

def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. uses 'torchensemble' library to ensemble model based on those data
    4. computes MAPE and plots graph
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    with mlflow.start_run(run_name="ensemble",nested=True) as ensemble_start:

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
        # store mlflow metrics/artifacts on temp file
        with tempfile.TemporaryDirectory(dir='./temp_files/') as ensemble_tmpdir: 

            global click_params
            click_params = ClickParams(kwargs)

            print("=================== Click Params ===================")
            print(vars(click_params))

            #  read csv files
            df = read_csvs(click_params)
            df_backup = df.copy()

            # split data in train/test/validation
            global train_data, test_data, val_data
            train_data, test_data, val_data = train_test_valid_split(df,click_params)

            # train_data.to_csv(f"{ensemble_tmpdir}/train_data.csv")
            # test_data.to_csv(f"{ensemble_tmpdir}/test_data.csv")
            # val_data.to_csv(f"{ensemble_tmpdir}/val_data.csv")

            # train model with hparams set to best_params of optuna 
            pred, actual = ensemble(load=False, save_dir=ensemble_tmpdir)
            pd.DataFrame(actual).to_csv(f"{ensemble_tmpdir}/actual_data.csv")
            pd.DataFrame(pred).to_csv(f"{ensemble_tmpdir}/pred_data.csv")

            # calculate metrics
            metrics = calculate_metrics(actual,pred,df_backup,click_params)

            # plot prediction/actual data on common axis system 
            cross_plot_actual_pred(df_backup, pred, actual, ensemble_tmpdir)

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_params(kwargs)
            mlflow.log_artifacts(ensemble_tmpdir, "ensemble_results")
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_id", ensemble_start.info.run_id)
            mlflow.set_tag('ensemble_model_uri', f'{ensemble_start.info.artifact_uri}/ensemble_model')
            mlflow.set_tag("stage", "ensemble")

if __name__ == '__main__':
    print("\n=========== Ensemble Model =============")
    logging.info("\n=========== Ensemble Model =============")
    forecasting_model()