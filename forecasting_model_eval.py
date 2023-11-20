import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import logging
import click
import mlflow
from mlflow.models.signature import infer_signature
import tempfile
import os

import torchensemble 

# Custom made classes and functions used for reading/processing data at input/output of model
from model_utils import ClickParams, Regression, Transfer
from model_utils import read_csvs, train_test_valid_split, \
                        feature_target_split, cross_plot_actual_pred, calculate_metrics
# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

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

def test_best_model(test_loader):

    ensemble_model = mlflow.pytorch.load_model(click_params.model_uri)

    # At the end of constructor. set feature/target of model
    # create datasets used by dataloaders
        # 'subset'_X: dataset containing features of subset (train/test/validation) dataframe
        # 'subset'_Y: dataset containing targets of subset (train/test/validation) dataframe

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

    # reshape 1-D matrix (created by list train_X) for it to be multipled by input layer matrix
    # matrix shape: [any, lookback_window]
    preds = ensemble_model.predict(np.array(actuals_X).reshape(-1, click_params.l_window))

    # store BaggingRegressor in mlflow as artifact "ensemble_model" with input singature
    # signature = infer_signature(train_X.head(1), pd.DataFrame(preds)) 
    mlflow.pytorch.log_model(ensemble_model, "ensemble_model")

    # convert preds to 1-D list
    # keep in actuals_Y, the targets of features used by predict()
    preds = [item for sublist in preds.tolist() for item in sublist]
    actuals_Y = actuals_Y[:len(preds)]

    return preds, actuals_Y

@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--dir_in", type=str, default='../preprocessed_data/', help="File containing csv files used by the model")
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option('--test_years', type=str, default='2021', help='list of years to use for testing set')
@click.option('--model_uri', type=str, default='None', help='model uri used')
@click.option("--l_window", type=str, default="240", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24", help='range of forecast horizon (output layer size) used by the model')
@click.option("--transfer_mode", type=str, default="0", help='indicator to use transfer learning techniques')
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
    with mlflow.start_run(run_name="eval",nested=True) as eval_start:

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
        # store mlflow metrics/artifacts on temp file
        with tempfile.TemporaryDirectory(dir='./temp_files/') as eval_tmpdir: 
            
            print(f'kwargs: {kwargs}')
            global click_params
            click_params = ClickParams(kwargs)
            print(f'click_params: {vars(click_params)}')

            #  read csv files
            df = read_csvs(click_params)
            df_backup = df.copy()

            # get test_data
            global test_data
            test_data = df[df['year'] == click_params.test_years][['Load']]
            test_data.to_csv(f"{eval_tmpdir}/test_data.csv")

            test_X, test_Y = feature_target_split(test_data,click_params.l_window,click_params.f_horizon)
            feature = torch.tensor(test_X.values).float()
            target = torch.tensor(test_Y.values).float()
            test_dataset = TensorDataset(feature, target)
            test_loader = DataLoader(dataset = test_dataset)

            # train model with hparams set to best_params of optuna 
            pred, actual = test_best_model(test_loader)
            pd.DataFrame(actual).to_csv(f"{eval_tmpdir}/actual_data.csv")
            pd.DataFrame(pred).to_csv(f"{eval_tmpdir}/pred_data.csv")
            
            # calculate metrics
            metrics = calculate_metrics(actual,pred,df_backup,click_params)

            # plot prediction/actual data on common axis system 
            cross_plot_actual_pred(df_backup, pred, actual, eval_tmpdir)

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_params(kwargs)
            mlflow.log_artifacts(eval_tmpdir, "eval_results")
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_id", eval_start.info.run_id)        
            mlflow.set_tag('ensemble_model_uri', f'{eval_start.info.artifact_uri}/ensemble_model')
            mlflow.set_tag("stage", "eval")
            mlflow.set_tag("transfer", Transfer(click_params.transfer_mode).name)

if __name__ == '__main__':
    print("\n=========== Eval Model =============")
    logging.info("\n=========== Eval Model =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)    
    forecasting_model()