#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)

# Let's import some basic libraries
import math
import pandas as pd 
from pandas.tseries.frequencies import to_offset
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("webagg")
import warnings
import copy
import tempfile

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import os
import logging
import pickle
import click
import mlflow

from model_utils import ClickParams, Regression
from model_utils import read_csvs, train_test_valid_split, feature_target_split

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
click_params = None
freq = None

"""
This function is use by the Optuna library: An open source hyperparameter optimization framework
to automate hyperparameter search
In Optuna, there are two major terminologies, namely: 
1) Study: The whole optimization process is based on an objective function 
   i.e the study needs a function which it can optimize. 
2) Trial: A single execution of the optimization function is called a trial. 
Thus the study is a collection of trials

Optuna is a black-box optimizer, which means it needs an objective function, 
which returns a numerical value to evaluate the performance of the hyperparameters, 
and decide where to sample in upcoming trials

In Optuna:
    We create a "study", which corresponds to an optimization task (maximize/minimize the objective function)
    We then try to optimise (maximise/minimize) that objective 
        Since our objective functions returns (and therefore is based on) validation loss, we want
        to minimize that value rather than maximize
    Each "execution" of the objective function is basically a "trial"
        trial (in context): one training of model for specific set of suggestive parameters (check params)
"""
def objective(trial):
    """
    Function used by optuna for hyperparameter tuning
    Each execution of this function is basically a new trial with its params
    changed as suggest by suggest_* commands
    Parameters: 
        trial object (default)
    Returns: 
        validation loss of model used for checking progress of tuning 
    """
    global click_params

    max_epochs = click_params.max_epochs #500 #trial.suggest_int('max_epochs', click_params.max_epochs[0], click_params.max_epochs[1])
    n_layers = trial.suggest_int("n_layers", click_params.n_layers[0], click_params.n_layers[1])

    timedelta_freq = pd.to_timedelta(to_offset(freq)) #convert freq to timedelta
    step = math.floor(pd.Timedelta(hours=24) / timedelta_freq) #how many freq fit in 24 hours 
    if(not step): step = 1 #if zero, set to one

    # if single item in categorical variables, turn it to a list
    if isinstance(click_params.optimizer_name, str): click_params.optimizer_name = [click_params.optimizer_name] 
    if isinstance(click_params.activation, str): click_params.activation = [click_params.activation]

    params = {
        'l_window': trial.suggest_categorical("l_window", click_params.l_window), 
        'f_horizon': 24, 
        'layer_sizes': [trial.suggest_categorical("n_units_l{}".format(i), click_params.layer_sizes) for i in range(n_layers)], 
        'l_rate':  trial.suggest_float('l_rate', click_params.l_rate[0], click_params.l_rate[1], log=True), # loguniform will become deprecated
        'activation': trial.suggest_categorical("activation", click_params.activation), #SiLU (Swish) performs good
        'optimizer_name': trial.suggest_categorical("optimizer_name", click_params.optimizer_name),
        'batch_size': trial.suggest_categorical('batch_size', click_params.batch_size),
        'num_workers': click_params.num_workers
    }

    print(params)

    # num_retries = 10
    # for attempt_no in range(num_retries):
    #     try:
    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    trainer = Trainer(max_epochs=max_epochs, deterministic=True, logger=True, 
                    # accelerator='auto', 
                    # devices = 1 if torch.cuda.is_available() else 0,
                    auto_select_gpus=True if torch.cuda.is_available() else False,
                    check_val_every_n_epoch=2,
                    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"), 
                                EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=10)])

    pl.seed_everything(click_params.seed, workers=True)    
    model = Regression(**params) # double asterisk (dictionary unpacking)
    trainer.logger.log_hyperparams(params)

    # At the end of constructor. set feauture/target of model
    # create datasets used by dataloaders
        # 'subset'_X: dataset containing features of subset (train/test/validation) dataframe
        # 'subset'_Y: dataset containing targets of subset (train/test/validation) dataframe
    train_X, train_Y = feature_target_split(train_data, params['l_window'], params['f_horizon'])  
    test_X, test_Y = feature_target_split(test_data, params['l_window'], params['f_horizon'])
    validation_X, validation_Y = feature_target_split(val_data, params['l_window'], params['f_horizon'])      

    train_loader = model.train_dataloader(train_X,train_Y)
    test_loader = model.test_dataloader(test_X,test_Y)
    val_loader = model.val_dataloader(validation_X,validation_Y)

    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_loss"].item()
                
def store_params(study, opt_tmpdir):
    best_params = {}; best_params.update(study.best_params)
    best_params['layer_sizes'] = ','.join(str(value) 
                                    for key,value in best_params.items() 
                                    if key.startswith('n_units_l'))

    # remove n_units_lXXX elements 
    best_params = { k: v for k, v in best_params.items() 
                    if not k.startswith("n_units_l")}

    print(f'Store best_params: {best_params}')
    # write binary, overwrite if file exists, creates file if not exists
    best_trial_file = open(f"{opt_tmpdir}/optuna_best_trial.pkl", "wb") 
    pickle.dump(best_params, best_trial_file)
    best_trial_file.close()    

    best_result = copy.deepcopy(study.best_params)
    best_result['value'] = study.best_trial.value

    # appends, pointer at EOF if file exists, creates file if not exists    
    with open('best_trial_diary.txt','a') as trial_diary_file: 
        trial_diary_file.write(str(best_result)+ "\n")

    with open(f"{opt_tmpdir}/best_trial.txt",'a') as trial_file:
        trial_file.write(f'========= Optuna Best Trial =========\n')
        for key, value in best_result.items():
            trial_file.write(f'{key}: {value}\n')
            
    study.trials_dataframe().to_csv(f"{opt_tmpdir}/trials_dataframe.csv")
                
def print_optuna_report(study):
    """
    This function prints hyperparameters found by optuna
    """    
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna Report ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# The pruners module defines a BasePruner class characterized by an abstract prune() method, which, 
# for a given trial and its associated study, returns a boolean value 
# representing whether the trial should be pruned (aborted)
# optuna.pruners.MedianPruner() 
# optuna.pruners.NopPruner() (no pruning)
# Hyperband performs best with default sampler for non-deep learning tasks
def optuna_optimize():
    """
    Function used to setup optuna for study
    Parameters: None
    Returns: study object containing info about trials
    """
    # The pruners module defines a BasePruner class characterized by an abstract prune() method, which, 
    # for a given trial and its associated study, returns a boolean value 
    # representing whether the trial should be pruned (aborted)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    # optuna.pruners.MedianPruner() 
    # optuna.pruners.NopPruner() (no pruning)
    # Hyperband performs best with default sampler for non-deep learning tasks
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner() # experimentally better performance
        
    # default sampler: TPMESampler    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    """
    timeout (Union[None, float]) – Stop study after the given number of second(s). 
    None represents no limit in terms of elapsed time. 
    The study continues to create trials until: the number of trials reaches n_trials, 
                                                timeout period elapses, stop() is called or, 
                                                a termination signal such as SIGTERM or Ctrl+C is received.
    """
    study.optimize(objective,
                #  n_jobs=2,
                #    timeout=600, # 10 minutes
                   n_trials=10, #click_params.n_trials,
                   gc_after_trial=True)
    print_optuna_report(study)
    return study

def optuna_visualize(study, opt_tmpdir):
    # if(optuna.visualization.is_available()):
    #     optuna.visualization.plot_param_importances(study).show()
    #     optuna.visualization.plot_optimization_history(study).show()
    #     optuna.visualization.plot_intermediate_values(study).show()
    #     optuna.visualization.plot_slice(study, study.best_params).show()
    #     optuna.visualization.plot_contour(study).show()

    plt.close() # close any mpl figures (important, doesn't work otherwise)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f"{opt_tmpdir}/plot_param_importances.png"); plt.close()

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f"{opt_tmpdir}/plot_optimization_history.png"); plt.close()
    
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(f"{opt_tmpdir}/plot_slice.png"); plt.close()

    optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.savefig(f"{opt_tmpdir}/plot_intermediate_values.png"); plt.close()

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
@click.option("--n_trials", type=str, default="20", help='number of trials - different tuning oh hyperparams')
@click.option("--max_epochs", type=str, default="5,10", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1,2", help='range of number of layers used by the model')
@click.option("--layer_sizes", type=str, default="90,110", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="220,260", help='range of lookback window (input layer size) used by the model')
@click.option("--l_rate", type=str, default="1e-5, 1e-4", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU,SiLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam,RMSprop", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="256,512,1024", help='possible batch sizes used by the model') #16,32,
@click.option("--num_workers", type=str, default="4", help='accelerator (cpu/gpu) processesors and threads used') 

def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. uses 'optuna' library to tune hyperparameters in range based on click params
    4. computes MAPE and plots graph of best model
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    with mlflow.start_run(run_name="optuna",nested=True) as optuna_start:
        # Auto log all MLflow entities
        # mlflow.pytorch.autolog()
        
        if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
        # store mlflow metrics/artifacts on temp file
        with tempfile.TemporaryDirectory(dir='./temp_files/') as opt_tmpdir:

            global click_params
            click_params = ClickParams(kwargs)

            print("=================== Click Params ===================")
            print(vars(click_params))
            
            #  read csv files
            df = read_csvs(click_params)
            df_backup = df.copy()

            df_backup.set_index('Date',inplace=True)
            global freq; freq = pd.infer_freq(df_backup.index) #find time frequency of data
            if(not freq): freq = 'H'

            # split data in train/test/validation
            global train_data, test_data, val_data
            train_data, test_data, val_data = train_test_valid_split(df,click_params)

            # train_data.to_csv(f"{opt_tmpdir}/train_data.csv")
            # test_data.to_csv(f"{opt_tmpdir}/test_data.csv")
            # val_data.to_csv(f"{opt_tmpdir}/val_data.csv")

            # use Optuna to make a study for best hyperparamter values for maxiumum reduction of loss function
            study = optuna_optimize()

            store_params(study, opt_tmpdir)        
            
            # visualize results of study
            optuna_visualize(study, opt_tmpdir)

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_params(kwargs)
            mlflow.log_artifacts(opt_tmpdir, "optuna_results")
            mlflow.set_tag("run_id", optuna_start.info.run_id)
            mlflow.set_tag('best_trial_uri', f'{optuna_start.info.artifact_uri}/optuna_results/optuna_best_trial.pkl')
            mlflow.set_tag("stage", "optuna")
        
if __name__ == '__main__':
    print("\n=========== Optuna Hyperparameter Tuning =============")
    logging.info("\n=========== Optuna Hyperparameter Tuning =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)    
    forecasting_model()