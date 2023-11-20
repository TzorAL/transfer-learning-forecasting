#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)
# 
# Let's import some basic libraries
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import logging
import click
import mlflow
from mlflow.models.signature import infer_signature
import tempfile
import shutil
import os

# Custom made classes and functions used for reading/processing data at input/output of model
from model_utils import ClickParams, Regression, Transfer
from model_utils import read_csvs, train_test_valid_split, \
                        feature_target_split, cross_plot_actual_pred, calculate_metrics

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
test_X,test_Y = None, None # used by MASE metric 

def train_test_best_model(train_tmpdir):
    """
    This function is used to train and test our pytorch lightning model
    based on parameters give to it
    Parameters: None (basically global click_params)
    Returns: 
        actuals: 1-D list containing target labels
        predictions:  1-D list containing predictions for said labels
    """
    params = vars(click_params)
    
    # layer_sizes must be iterable for creating model layers
    if(not isinstance(params['layer_sizes'],list)):
        params['layer_sizes'] = [params['layer_sizes']] 

    torch.set_num_threads(click_params.num_workers) #################################
    pl.seed_everything(click_params.seed, workers=True)  

    model = Regression(**params) # double asterisk (dictionary unpacking)

    max_epochs = None
    max_epochs = 5 if(model.transfer_mode == Transfer.BOUNDED_EPOCHS) \
                   else params['max_epochs']

    trainer = Trainer(max_epochs=max_epochs, deterministic=True,
                      profiler=SimpleProfiler(dirpath=f'{train_tmpdir}/profiler_reports/', filename='profiler_report'), #add simple profiler
                      logger= CSVLogger(save_dir=train_tmpdir),
                    #   accelerator='auto', 
                    #   devices = 1 if torch.cuda.is_available() else 0,
                      auto_select_gpus=True if torch.cuda.is_available() else False,
                      check_val_every_n_epoch=2,
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=10)]) 

    # At the end of constructor. set feauture/target of model
    # create datasets used by dataloaders
        # 'subset'_X: dataset containing features of subset (train/test/validation) dataframe
        # 'subset'_Y: dataset containing targets of subset (train/test/validation) dataframe
    global test_X,test_Y,validation_X # used by MASE metric 
    train_X, train_Y = feature_target_split(train_data,click_params.l_window,click_params.f_horizon)  
    test_X, test_Y = feature_target_split(test_data,click_params.l_window,click_params.f_horizon)
    validation_X, validation_Y = feature_target_split(val_data, click_params.l_window,click_params.f_horizon)      

    train_loader = model.train_dataloader(train_X,train_Y)
    test_loader = model.test_dataloader(test_X,test_Y)
    val_loader = model.val_dataloader(validation_X,validation_Y)

    trainer.fit(model, train_loader, val_loader)

    pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
    
    # Either best or path to the checkpoint you wish to test. 
    # If None and the model instance was passed, use the current weights. 
    # Otherwise, the best model from the previous trainer.fit call will be loaded.
    trainer.test(ckpt_path='best', dataloaders=test_loader)

    trainer.validate(ckpt_path='best', dataloaders=val_loader)

    preds = trainer.predict(ckpt_path='best', dataloaders=test_loader)

    #store trained model to mlflow with input singature
    signature = infer_signature(train_X.head(1), pd.DataFrame(preds))
    mlflow.pytorch.log_model(model, "model", signature=signature)

    actuals = []
    for x,y in test_loader: 
        actuals.append(y)

    # torch.vstack: convert list of tensors to (rank 2) tensor
    # .tolist(): convert (rank 2) tensor to list of lists
    # final outcome: list of floats
    preds = [item for sublist in torch.vstack(preds).tolist() for item in sublist]
    actuals = [item for sublist in torch.vstack(actuals).tolist() for item in sublist]

    return preds, actuals

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
@click.option("--max_epochs", type=str, default="200", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1", help='range of number of layers used by the model')
@click.option("--layer_sizes", type=str, default="100", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="240", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24", help='range of forecast horizon (output layer size) used by the model')
@click.option("--l_rate", type=str, default="1e-4", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="1024", help='possible batch sizes used by the model') #16,32,
@click.option("--transfer_mode", type=str, default="0", help='indicator to use transfer learning techniques')
@click.option("--num_workers", type=str, default="4", help='accelerator (cpu/gpu) processesors and threads used') 
@click.option("--tl_model_uri", type=str, default="None", help='tl_model_uri used for accessing model used for transfer learning') 
@click.option('--time_steps', type=str, default='168', help='naive model time lags')

def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. trains model based on hyper params set
    4. computes MAPE and plots graph of best model
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    with mlflow.start_run(run_name="train",nested=True) as train_start:

        # Auto log all MLflow entities
        # mlflow.pytorch.autolog()

        if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
        # store mlflow metrics/artifacts on temp file
        with tempfile.TemporaryDirectory(dir='./temp_files/') as train_tmpdir: 

            global click_params
            click_params = ClickParams(kwargs)
            
            # #  read csv files
            df = read_csvs(click_params)
            df_backup = df.copy()

            # split data in train/test/validation
            global train_data, test_data, val_data
            train_data, test_data, val_data = train_test_valid_split(df,click_params)

            # train_data.to_csv(f"{train_tmpdir}/train_data.csv")
            # test_data.to_csv(f"{train_tmpdir}/test_data.csv")
            # val_data.to_csv(f"{train_tmpdir}/val_data.csv")

            # train model with hparams set to best_params of optuna 
            pred, actual = train_test_best_model(train_tmpdir)

            # calculate metrics
            metrics = calculate_metrics(actual,pred,df_backup)

            # plot prediction/actual data on common axis system 
            cross_plot_actual_pred(df_backup, pred, actual, train_tmpdir)

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_params(kwargs)
            mlflow.log_artifacts(train_tmpdir, "train_results")
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_id", train_start.info.run_id)        
            mlflow.set_tag('tl_model_uri', f"runs:/{train_start.info.run_id}/model")
            mlflow.set_tag("stage", "model")
            mlflow.set_tag("transfer", Transfer(click_params.transfer_mode).name)


if __name__ == '__main__':
    print("\n=========== Forecasing Model =============")
    logging.info("\n=========== Forecasing Model =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)    
    forecasting_model()