import numpy as np
import pandas as pd
from enum import IntEnum
from datetime import datetime
import pathlib
from matplotlib import pyplot as plt
import random
# import matplotlib as mpl
# mpl.use("webagg")
import warnings

import sys
import copy
import os
import gc
import tempfile
import requests

from sklearn import preprocessing

from darts import TimeSeries
from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.metrics import mse as mse_darts

from darts.models import NaiveSeasonal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
click_params = None
train_data = None
test_data = None 
val_data = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
#### Enum used to define transfer learning techniques used  
class Transfer(IntEnum):
    NO_TRANSFER = 0
    WARM_START = 1 
    BOUNDED_EPOCHS = 2
    FREEZING = 3
    HEAD_REPLACEMENT = 4

#### Enum used to define test case used in run of test_script  
class TestCase(IntEnum):
    NAIVE = 0
    BENCHMARK = 1
    ALL_FOR_ONE = 2 
    CLUSTER_FOR_ONE = 3
    GLOBAL = 4

#### Class used to store and utilize click arguments passed to the script
class ClickParams:
    def __init__(self, click_args):
        super(ClickParams, self).__init__()
        
        for key, value in click_args.items():
            if(isinstance(value, bool)):
               setattr(self, key, value); continue 
            if(os.sep in value): # differentiate between dir_in and optimizer/activation
               setattr(self, key, value); continue 
     
            split_value = value.split(',') # split string into a list delimited by comma

            if not split_value: # if empty list
                sys.exit(f"Empty value for parameter {key}")
            
            if(len(split_value) == 1): # if single value, check if int/float/str
                if(value.isdigit()): setattr(self, key, int(value)); continue;
                if(self.is_floats(value)): setattr(self, key, float(value)); continue;
                else: setattr(self, key, value); continue; # categorical should be kept as a list  

            # since value is a list
            if all(ele.isdigit() for ele in split_value): # if all characters in each value in list are digits (int)
                setattr(self, key, [int(x) for x in split_value]); continue;
            elif self.is_floats(split_value): # check if all values in list can be converted to floats (decimals or scientific notation)
                setattr(self, key, [float(x) for x in split_value]); continue;
            else: # then it's a list of strings
                setattr(self, key, split_value)
    
    def is_floats(self,x):
        """ True if x is convertible to a float, or is a list or tuple (or nested list or 
            tuple) of values that are all convertible to floats """    
        if isinstance(x, (list, tuple)):
            return all(self.is_floats(item) for item in x)
        try:
            float(x)
            return True
        except ValueError:
            return False

def download_online_file(url, dst_filename=None, dst_dir=None):

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception(f"\nResponse is not 200\nProblem downloading: {url}")
        sys.exit()
    url_content = req.content
    if dst_filename is None:
        dst_filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, dst_filename)
    file = open(filepath, 'wb')
    file.write(url_content)
    file.close()

    return filepath

def download_mlflow_file(url, dst_dir=None):
    S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    if url.startswith('s3://mlflow-bucket/'):
        url = url.replace("s3:/", S3_ENDPOINT_URL)
        local_path = download_online_file(
            url, dst_dir=dst_dir)
    elif url.startswith('runs:/'):
        # client = mlflow.tracking.MlflowClient()
        run_id = url.split('/')[1]
        mlflow_path = '/'.join(url.split('/')[3:])
        # local_path = client.download_artifacts(run_id, mlflow_path, dst_dir)
        local_path = mlflow.artifacts.download_artifacts(run_id, mlflow_path, dst_dir)
    elif url.startswith('http://'):
        local_path = download_online_file(url, dst_dir=dst_dir)
    elif url.startswith('file:///'):
        mlflow_path = '/'.join(url.split('/')[3:])
        local_path = mlflow.artifacts.download_artifacts(mlflow_path, dst_path=dst_dir)
    else:
        return url

    return local_path

def search_proper_run():
    # find best pretrained model by searching runs with
    # search only runs in 'model' stage and without transfer learning
    filter_string = f'tags.stage = "model" and tags.transfer = "NO_TRANSFER"' 
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.get_run(run_id).info.experiment_id

    best_run = mlflow.search_runs(experiment_ids=[experiment_id],# experiment_names=["alex_trash"],
                                  output_format='list',
                                  filter_string=filter_string,
                                  run_view_type=ViewType.ACTIVE_ONLY,
                                  max_results=1,
                                # order_by=["metrics.MAPE DESC"]
                                  order_by=['attribute.end_time DESC']
                                  )[0]
    print(f"Found best model run with ID: {best_run.info.run_id}")
    return best_run

#### Define the model and hyperparameters
class Regression(pl.LightningModule):
    """
    Regression  Techniques are used when the output is real-valued based on continuous variables. 
                For example, any time series data. This technique involves fitting a line
    Feature: Features are individual independent variables that act as the input in your system. 
             Prediction models use features to make predictions. 
             New features can also be obtained from old features using a method known as ‘feature engineering’. 
             More simply, you can consider one column of your data set to be one feature. 
             Sometimes these are also called attributes. T
             The number of features are called dimensions
    Target: The target is whatever the output of the input variables. 
            In our case, it is the output value range of load. 
            If the training set is considered then the target is the training output values that will be considered.
    Labels: Label: Labels are the final output. You can also consider the output classes to be the labels. 
            When data scientists speak of labeled data, they mean groups of samples that have been tagged to one or more labels.

    ### The Model ### 
    Initialize the layers
    Here we have:
        one input layer (size 'lookback_window'), 
        one output layer (size 36 as we are predicting next 36 hours)
        hidden layers define by 'params' argument of init
    """
    def __init__(self, **params):
        super(Regression, self).__init__()

        # used by trainer logger (check log_graph flag)
        # example of input use by model (random tensor of same size)
        self.example_input_array = torch.rand(params['l_window'])

        self.loss = nn.MSELoss()

        # by default, no transfer learning, init in case of optuna/ensemble        
        self.transfer_mode = Transfer(params['transfer_mode']) if('transfer_mode'in params) \
                             else Transfer.NO_TRANSFER 

        # enable Lightning to store all the provided arguments 
        # under the self.hparams attribute. 
        # These hyperparameters will also be stored within the model checkpoint
        self.save_hyperparameters()

        #input dim set to lookback_window while output dim set to f_horizon
        input_dim, output_dim = self.hparams.l_window, self.hparams.f_horizon

        """
        feature_extractor: all layers before classifier
        classifier: last layer connecting output with rest of network (not always directly)
        We load proper pretrained model, and use its feauture_extractor for the new untrained one
        (Also check forward pass commentary)
        """
        self.feature_extractor = None        
        self.classifier = None

        if(self.transfer_mode == Transfer.NO_TRANSFER):
            feature_layers, last_dim = self.make_hidden_layers()
            self.feature_extractor = nn.Sequential(*feature_layers) #list of nn layers
            self.classifier = nn.Linear(last_dim, output_dim)
        else:
            
            model = mlflow.pytorch.load_model(self.hparams.tl_model_uri)
            print(f"Found best model with uri: {self.hparams.tl_model_uri}")
            mlflow.set_tag("tl_model_used", self.hparams.tl_model_uri)

            # get dimension of last hidden layer of model (one before output)
            last_dim = model.hparams.layer_sizes[-1]
            self.feature_extractor = model.feature_extractor # "old" feature extractor
            self.classifier = nn.Linear(last_dim, output_dim) # new classifier

            if(self.transfer_mode == Transfer.WARM_START):
                print('Using \"WARM START\" technique...')
                self.classifier = model.classifier # "old" classifier

            # if(self.transfer_mode == Transfer.FREEZING):
            #     print('Using \"FREEZING\" technique...') # freeze feature layers
            #     # self.feature_extractor.eval() # disable dropout/BatchNorm layers using eval()
            #     self.feature_extractor.requires_grad_(False) # freeze params
                # self.feature_extractor.freeze()     

    def make_hidden_layers(self):
        """
        Each loop is the setup of a new layer
        At each iteration:
            1. add previous layer to the next (with parameters gotten from layer_sizes)
                    at first iteration previous layer is input layer
            2. add activation function
            3. set current_layer as next layer
        connect last layer with cur_layer
        Parameters: None
        Returns: 
            layers: list containing input layer through last hidden one
            cur_layer: size (dimension) of the last hidden layer      
        """
        layers = [] # list of layer to add at nn
        cur_layer = self.hparams.l_window

        for next_layer in self.hparams.layer_sizes: 
            print(f'({cur_layer},{next_layer},{self.hparams.layer_sizes})')
            layers.append(nn.Linear(cur_layer, next_layer))
            layers.append(getattr(nn, self.hparams.activation)()) # nn.activation_function (as suggested by Optuna)
            cur_layer = next_layer #connect cur_layer with previous layer (at first iter, input layer)
        return layers, cur_layer

    # Perform the forward pass
    def forward(self, x):
        """
        In forward pass, we pass input through (freezed or not) feauture extractor
        and then its output through the classifier 
        """
        representations = self.feature_extractor(x)
        return self.classifier(representations)

### The Data Loaders ###     
    # Define functions for data loading: train / validate / test

# If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, 
# you can speed up the host to device transfer by enabling "pin_memory".
# This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.    
    def train_dataloader(self,train_X, train_Y):
        feature = torch.tensor(train_X.values).float() #feature tensor train_X
        target = torch.tensor(train_Y.values).float() #target tensor train_Y
        train_dataset = TensorDataset(feature, target)  # dataset bassed on feature/target
        train_loader = DataLoader(dataset = train_dataset, 
                                  shuffle = True, 
                                  pin_memory=True if torch.cuda.is_available() else False, #for GPU
                                  num_workers = self.hparams.num_workers,
                                  batch_size = self.hparams.batch_size)
        return train_loader
            
    def test_dataloader(self,test_X,test_Y):
        feature = torch.tensor(test_X.values).float()
        target = torch.tensor(test_Y.values).float()
        test_dataset = TensorDataset(feature, target)
        test_loader = DataLoader(dataset = test_dataset, 
                                 pin_memory=True if torch.cuda.is_available() else False, #for GPU
                                 num_workers = self.hparams.num_workers,
                                 batch_size = self.hparams.batch_size)
        return test_loader

    def val_dataloader(self,validation_X,validation_Y):
        feature = torch.tensor(validation_X.values).float()
        target = torch.tensor(validation_Y.values).float()
        val_dataset = TensorDataset(feature, target)
        validation_loader = DataLoader(dataset = val_dataset,
                                       pin_memory=True if torch.cuda.is_available() else False, #for GPU
                                       num_workers = self.hparams.num_workers,
                                       batch_size = self.hparams.batch_size)
        return validation_loader

    def predict_dataloader(self):
        return self.test_dataloader()
    
### The Optimizer ### 
    # Define optimizer function: here we are using ADAM
    def configure_optimizers(self):
        return getattr(optim, self.hparams.optimizer_name)( self.parameters(),
                                                            # momentum=0.9, 
                                                            # weight_decay=1e-4,                   
                                                            lr=self.hparams.l_rate)

### Training ### 
    # Define training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        self.log("loss", loss, on_epoch=True) # computes train_loss mean at end of epoch        
        return {'loss': loss, 'log': logs}

### Validation ###  
    # Define validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        self.log("avg_val_loss", loss, on_epoch=True)  # computes avg_loss mean at end of epoch
        return {'val_loss': loss}

### Testing ###     
    # Define test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        correct = torch.sum(logits == y.data)
        # I want to visualize my predictions vs my actuals so here I'm going to 
        # add these lines to extract the data for plotting later on
        self.log('test_loss', loss, on_epoch=True)        
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}

### Prediction ###
    # Define prediction step
        # This method takes as input a single batch of data and makes predictions on it. 
        # It then returns predictions
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.forward(x)
        
    def on_train_epoch_end(self):
        gc.collect()

    def on_validation_epoch_end(self):
        gc.collect()

## Loading Data
def read_csvs(click_params):
    """
    Reads all csv files in folder path given and stores them in a single dataframe df 
    Parameters: list of countries to read their csv
    ----------
    Returns
    -------
    pandas.DataFrame
        Dataframe where rows are individual data and columns for indicative
        information on the data like datetime, month, week, country etc
    """

    df = pd.DataFrame()
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # For every csv file in folder 
    for csv in pathlib.Path(click_params.dir_in).glob('*.csv'):
        if(click_params.countries is None or csv.stem in click_params.countries):            
            print("Loading dataset: "+csv.stem)
            temp_df = pd.read_csv(csv,
                                  parse_dates=['Date'],
                                  dayfirst=True,
                                  date_parser=dateparse)
            temp_df['country'] = csv.stem #set cokumn with the name of the dataset this datum came from
            df = pd.concat([df,temp_df]) 
    print('Data loading complete!')
    # sort values based on Start column (Datetime)
    df = df[df['year'] != 2022] # remove rows of year 2022 since its an incomplete year (not currently ended)
    if(click_params.countries is None):
        df.sort_values(by=['Date'], ascending=True, inplace=True)
    df['Unnamed: 0'] = range(1, len(df) + 1)     # Reset "Unnamed: 0" based on concated df 
    return df

def train_test_valid_split(df,click_params):
    """
    we choose to split data with validation/test data to be at the end of time series
    Since it's a time series, it is intuitive to predict future values, rather than values in between
    validation/test should hold at least an entire year, since:
        - most seasonality exists between years
        - there are fluctuation inside year (e.g heatwaves at summer)
    Parameters:
        pandas.dataframe containing dataframe to split
    Returns:
        pandas.dataframe containing train/test/valiation data
        pandas.dataframe containing valiation data
        pandas.dataframe containing test data
    """
    # train_data: data from year [2015,2019]
    # validation_data: data from year 2020
    # test_data: data from year 2021
    # drop all columns except 'Load' (use 'backup_df' to restore them)
    global train_data, test_data, val_data


    train_data = df[~df['year'].isin([click_params.val_years,click_params.test_years])][['Load']] 
    test_data = df[df['year'] == click_params.test_years][['Load']]
    val_data = df[df['year'] == click_params.val_years][['Load']]

    return train_data, test_data, val_data

def feature_target_split(df, lookback_window=168, forecast_horizon=36):# lookback_window: 168 = 7 days(* 24 hours)
    """
    This function gets a column of a dataframe and splits it to input and target
    
    **lookback_window**
    In a for-loop of 'lookback_window' max iterations, starting from 0 
    At N-th iteration (iter): 
        1. create a shifted version of 'Load' column by N rows (vertically) and 
        2. stores it in a column* (feature_'N')

    Same pseudo-code for 'forecast_horizon' loop
    
    *At first iterations of both loops, the new columns (feature/target) are going to be firstly created
    but for each iteration, the same columns are going to be used
    
    We store each new column created in a dictionary which, at the end, convert it to dataframe
    The reason behind this choice is that if it was initially a dataframe, for large amount of loops,
    fast insertion of new columns would cause a performance issue (slower) even though the end result
    would propably not be altered
    
    Parameters: 
        df: pandas.dataframe containing column to parse
        lookback_window: lookback_window - # feature columns - # inputs in model
        forecast_horizon: forecast_horizon - # target columns - # outputs in model
    ----------
    Returns
        'subset'_X: pandas.dataframe containing features of df after preprocess for model
        'subset'_Y: pandas.dataframe containing targets of df after preprocess for model
    -------
    """

    # Reset "Unnamed: 0" based on concated df 
    df['Unnamed: 0'] = range(1, len(df) + 1)

    df_copy = df.copy()    

    df_new = {}
        
    for inc in range(0,int(lookback_window)):
        df_new['feature_' + str(inc)] = df_copy['Load'].shift(-inc)

    # shift 'load' column permanently for as many shifts happened at 'lookback_window' loops  
    df_copy['Load'] = df_copy['Load'].shift(-int(lookback_window))
                    
    for inc in range(0,int(forecast_horizon)):
        df_new['target_' + str(inc)] = df_copy['Load'].shift(-inc)    
    
    df_new = pd.DataFrame(df_new, index=df_copy.index)
    df_new = df_new.dropna().reset_index(drop=True)    
                            
    return df_new.iloc[:,:lookback_window] , df_new.iloc[:,-forecast_horizon:]

def cross_plot_actual_pred(df_backup, plot_pred_r, plot_actual_r, tmpdir):
    """
    Function used to plot lists of target labels and their predictions from the model
    in a shared axis system
    Parameters:
        df_backup: backup dataframe taken when reading csvs
        plot_actual: list containing target labels
        plot_pred: list containing predictions for said labels
    Returns: None
    """
    warnings.filterwarnings("ignore")

    # Get dates for plotting (test_data)
    datesx = list(df_backup[df_backup['year'] == 2021]['Date'])

    # And finally we can see that our network has done a decent job of estimating!
    fig, ax = plt.subplots(figsize=(16,4))
    
    # convert list of lists to list (np.hstack)
    # get every N-th value of predicion and data liiiiiist and display them in common plot
    N = 1000
    ax.plot(plot_pred_r[::N], label='Prediction Data')
    ax.plot(plot_actual_r[::N], label='Actual Data')
    ax.set_xticklabels(datesx[::N], rotation=45)
    ax.legend();

    plt.savefig(f"{tmpdir}/cross_plot.png")
    plt.close()

def convert_list_to_timeseries(data,dates,row_dim):
    listed_data = [data[i: i+row_dim] for i in range(0, len(data), row_dim)]

    column_names = [f"Data_{i}" for i in range(0,row_dim)]
    series_df = pd.DataFrame(data=listed_data,columns=column_names)
    value_cols = list(series_df.columns)
    series_df['Dates'] = dates[:len(series_df)]

    print(f'Dataframe shape: {series_df.shape}')
    series =  TimeSeries.from_dataframe(series_df, time_col='Dates', value_cols=value_cols)
    return series

def calculate_metrics(actual,pred,df_backup,click_params):
    """
    Function used to calculate forecasting metrics
    Parameters:
        actual: pd.Dataframe containing actual labels 
        pred: pd.Dataframe containing predicted labels 
        df_backup: backup dataframe to filter datetime for our actual/predicted labels
        click_params: user parameters to determine range of test set in years 
    Returns: dictionary containing values from different metrics
    """
    train_df = df_backup[~df_backup['year'].isin([click_params.test_years])][['Date','Load']] 
    test_df = df_backup[df_backup['year'] == click_params.test_years][['Date','Load']]

    test_df = test_df.sort_values('Date').drop_duplicates('Date',keep='first')
    train_df = train_df.sort_values('Date').drop_duplicates('Date',keep='first')

    # make dataframe an replace initial load data with processed/prediction data
    actual_df = test_df[['Date']].copy(); actual_df['Load'] = pd.DataFrame(actual)
    pred_df = test_df[['Date']].copy(); pred_df['Load'] = pd.DataFrame(pred)

    # actual_df.to_csv('actual.csv')
    # pred_df.to_csv('pred.csv')
    # train_df.to_csv('train.csv')

    # convert dataframes to darts timeseries
    train_series = TimeSeries.from_dataframe(train_df, time_col='Date', value_cols='Load')
    actual_series = TimeSeries.from_dataframe(actual_df, time_col='Date', value_cols='Load')
    pred_series = TimeSeries.from_dataframe(pred_df, time_col='Date', value_cols='Load')

    # print(train_series)
    print(actual_series)
    print(pred_series)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
    # Evaluate the model prediction
    metrics = {
        "MAPE": mape_darts(actual_series,pred_series),
        "SMAPE": smape_darts(actual_series,pred_series),
        "MASE": mase_darts(actual_series,pred_series,train_series,m=click_params.time_steps),
        "MAE": mae_darts(actual_series,pred_series),
        "MSE": mse_darts(actual_series,pred_series),
        "RMSE": rmse_darts(actual_series,pred_series)
    }
    
    print("  Metrics: ")
    for key, value in metrics.items():
        print("    {}: {}".format(key, value))

    return metrics
