import numpy as np
import pandas as pd
import logging
import mlflow
# from mlflow.models.signature import infer_signature

import click
import tempfile
# import shutil
import os
import pickle
# Custom made classes and functions used for reading/processing data at input/output of model
from model_utils import ClickParams
from model_utils import read_csvs, train_test_valid_split
from darts.models import NaiveSeasonal
from darts import TimeSeries

from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.metrics import mse as mse_darts

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
test_X,test_Y = None, None # used by MASE metric 

def train_test_naive_model(df, train_tmpdir):

    # split data to train and test (and validation but not needed)
    test_data = df[df['year'] == click_params.test_years][['Date','Load']]
    test_series = TimeSeries.from_dataframe(test_data, time_col='Date', value_cols='Load')

    df_series = TimeSeries.from_dataframe(df, time_col='Date', value_cols='Load')

    # get earliest date of test series
    train_end = test_data['Date'].min()
    test_start = train_end - pd.Timedelta(click_params.time_steps-2, unit='h')

    # create model and create forecasts
    model = NaiveSeasonal(K = click_params.time_steps)
    forecast_series = model.historical_forecasts(df_series,
                                                 start=test_start,
                                                 forecast_horizon=click_params.time_steps,
                                                 verbose=True,
                                                 stride=1)
    
    #store trained model to mlflow with input singature
    pickle.dump(model, open(f"{train_tmpdir}/snaive_model.pkl", "wb"))

    # Evaluate the model prediction
    metrics = {
        "MAPE": mape_darts(test_series, forecast_series),
        "SMAPE": smape_darts(test_series, forecast_series),
        "MASE": mase_darts(test_series, forecast_series, df_series[:train_end], m=click_params.time_steps),
        "MAE": mae_darts(test_series, forecast_series),
        "MSE": mse_darts(test_series, forecast_series),
        "RMSE": rmse_darts(test_series, forecast_series)
    }

    print(metrics)

    return metrics

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--dir_in", type=str, default='../preprocessed_data/', help="File containing csv files used by the model")
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option("--tgt_country", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option('--train_years', type=str, default='2015,2016,2017,2018,2019', help='list of years to use for training set')
@click.option('--val_years', type=str, default='2020', help='list of years to use for validation set')
@click.option('--test_years', type=str, default='2021', help='list of years to use for testing set')
@click.option('--time_steps', type=str, default='168', help='naive model time lags')


def naive_forecast(**kwargs):
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
    with mlflow.start_run(run_name="snaive", nested=True) as train_start:

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

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
            train_data, test_data, val_data = train_test_valid_split(df, click_params)

            # train_data.to_csv(f"{train_tmpdir}/train_data.csv")
            # test_data.to_csv(f"{train_tmpdir}/test_data.csv")
            # val_data.to_csv(f"{train_tmpdir}/val_data.csv")

            # train model with hparams set to best_params of optuna 
            metrics = train_test_naive_model(df, train_tmpdir)

            # calculate metrics
            # metrics = calculate_metrics(actual,pred,df_backup,test_X,len(test_Y.columns))

            # plot prediction/actual data on common axis system 
            # cross_plot_actual_pred(df_backup, pred, actual, train_tmpdir)

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_params(kwargs)
            mlflow.log_artifacts(train_tmpdir, "train_results")
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_id", train_start.info.run_id)        
            mlflow.set_tag('tl_model_uri', f"runs:/{train_start.info.run_id}/model")
            mlflow.set_tag("stage", "naive")
            # mlflow.set_tag("transfer", Transfer(click_params.transfer_mode).name)

if __name__ == '__main__':
    print("\n=========== Naive Forecast =============")
    logging.info("\n=========== Naive Forecast =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    naive_forecast()