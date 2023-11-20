#!/usr/bin/env python
# coding: utf-8

import os
import click
import tempfile
import matplotlib.pyplot as plt
import tempfile
import shutil
import pickle
import logging 
import copy

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id

from model_utils import ClickParams, Transfer, TestCase
from model_utils import search_proper_run

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

tl_model_uri = None
ensemble_model_uri = None

def print_run_info(runs):
    for r in runs:
        print("run_id: {}".format(r.info.run_id))
        print("lifecycle_stage: {}".format(r.info.lifecycle_stage))
        print("metrics: {}".format(r.data.metrics))
        print("Status: {}".format(r.info.to_proto().status))
        # Exclude mlflow system tags
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        print("tags: {}".format(tags))

def _already_ran(entry_point_name, parameters, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = MlflowClient()
    all_run_infos = reversed(client.search_runs(experiment_id))

    for run_info in all_run_infos:
        full_run = client.get_run(run_info.info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue
        
        # if failed runs exist delete them and their folder
        if run_info.info.to_proto().status == RunStatus.FAILED:
            mlflow.delete_run(run_info.info.run_id) 
            print(f"Delete file: {run_info.info.run_id}")
            shutil.rmtree(f"./mlruns/0/{run_info.info.run_id}", ignore_errors=True)

        if run_info.info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping (run_id=%s, status=%s)")
                % (run_info.info.run_id, run_info.status)
            )
            continue

        return client.get_run(run_info.info.run_id)
    eprint("No matching run has been found.")
    return None

# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, ignore_previous_run=False, use_cache=True):
# TODO: this was removed to always run the pipeline from the beginning.
    # if not ignore_previous_run:
    #     existing_run = _already_ran(entrypoint, parameters)
    #     existing_run = None
    #     if use_cache and existing_run:
    #         # print(f'Found existing run (with ID=%s) for entrypoint=%s and parameters=%s')
    #         logging.warning(f"Found existing run (with ID={existing_run.info.run_id}) for entrypoint={entrypoint} and parameters={parameters}")
    #         return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint,
                                parameters=parameters,
                                env_manager="local")
    return MlflowClient().get_run(submitted_run.run_id)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_step(train_kwargs, pipeline_name):
    load_run = None; load_params = None; load_data_path = None
    print(f"\n=========== {pipeline_name}: Load Data Stage =============")
    logging.info(f"\n=========== {pipeline_name}: Load Data Stage =============")    
    load_params = {
        'dir_in':  train_kwargs['dir_in'],
        'countries': train_kwargs['countries']         
    }
    load_run = _get_or_run("load", parameters=load_params)
    load_data_path = mlflow.artifacts.download_artifacts(f'runs:/{load_run.info.run_id}/load_data')
    print(f'load_data_path : {load_data_path}')
    
    return load_run, load_data_path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ETL entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def etl_step(train_kwargs, load_run, load_data_path, pipeline_name):
    etl_run = None; etl_params = None; preprocessed_data_path = None
    if('etl' in train_kwargs['stages'] or 'all' in train_kwargs['stages']):
        print(f"\n=========== {pipeline_name}: ETL Stage =============")
        logging.info(f"\n=========== {pipeline_name}: ETL Stage =============")
        etl_params = {
            'dir_in':  load_data_path if load_run else train_kwargs['dir_in'],
            'countries': train_kwargs['countries'], 
            'local_tz': train_kwargs['local_tz']
        }
        print(etl_params)
        etl_run = _get_or_run("etl", parameters=etl_params)

        load_data_uri = load_run.data.tags['load_data_uri'].replace("s3:/", S3_ENDPOINT_URL)

        # fetch output folder containing preprocessed datasets from etl, to be used as input for the following entrypoints
        preprocessed_data_path = mlflow.artifacts.download_artifacts(f'runs:/{etl_run.info.run_id}/preprocessed_data')
        print(f'preprocessed_data_path : {preprocessed_data_path}')

        # 'dir_in' in the context of etl is original data folder, while 
        # in the context of everything  else, is the output of dir_in
        train_kwargs['dir_in'] = preprocessed_data_path

    return etl_run 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def optuna_step(train_kwargs, pipeline_name):
    optuna_run = None; optuna_params = None
    if(pipeline_name == 'TARGET'): #optuna is unecessary in target model
        return optuna_run, optuna_params

    if ('optuna' in train_kwargs['stages'] or 'all' in train_kwargs['stages']):
        print(f"\n=========== {pipeline_name}: Optuna Stage =============")
        logging.info(f"\n=========== {pipeline_name}: Optuna Stage =============")            
        # if max_epochs int and not list (range), then main's click params (default or manually set)
        #    are meant for 'model' entrypoint and therefore should be ignored by optuna  
        # if not, it means that main's click params should be passed to optuna run
        if(not isinstance(train_kwargs['max_epochs'], list)):
            optuna_params = {
                'dir_in': train_kwargs['dir_in'],
                'countries': train_kwargs['countries'],
                'seed': train_kwargs['seed'],
                'n_trials': train_kwargs['n_trials']
            }
            print(f'optuna_params: {optuna_params}')
            optuna_run = _get_or_run("optuna", parameters=optuna_params)   
        else:
            exclusion_list = {'stages','local_tz','transfer_mode','n_estimators'}
            optuna_params = {x: train_kwargs[x] for x in train_kwargs if x not in exclusion_list}
            print(f'optuna_params: {optuna_params}')
            optuna_run = _get_or_run("optuna", parameters=optuna_params)   

    return optuna_run, optuna_params

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_step(train_kwargs, pipeline_name, optuna_run):
    train_run = None; train_params = None; best_run = None
    global tl_model_uri

    if 'model' in train_kwargs['stages'] or 'all' in train_kwargs['stages']:
        print(f"\n=========== {pipeline_name}: Model Stage =============")
        logging.info(f"\n=========== {pipeline_name}: Model Stage =============")
        if(pipeline_name == 'SOURCE'):
            if(optuna_run):
                print("\n########### Optuna used for model ###########")
                logging.info("\n########### Optuna used for model ###########")
                best_trial_path = f'runs:/{optuna_run.info.run_id}/optuna_results/optuna_best_trial.pkl'
                train_params = mlflow.artifacts.download_artifacts(best_trial_path)
                train_params = pickle.load( open(train_params, 'rb') )

                train_params['dir_in'] = train_kwargs['dir_in'] 
                train_params['transfer_mode'] = train_kwargs['transfer_mode']
                train_params['countries'] = train_kwargs['countries']
                train_params['max_epochs'] = train_kwargs['max_epochs']
                train_params['tl_model_uri'] = train_kwargs['tl_model_uri']

                print(f'train_params: {train_params}')
                train_run = _get_or_run("model", parameters=train_params) 
            else:   
                print("\n########### Custom hyperparams used for model ###########")
                logging.info("\n########### Custom hyperparams used for model ###########")
                exclusion_list = {'stages','local_tz','n_estimators','n_trials'}
                train_params = {x: train_kwargs[x] for x in train_kwargs if x not in exclusion_list}

                print(f'train_params: {train_params}')
                train_run = _get_or_run("model", parameters=train_params) 
            
            # store as global the transfer learning model uri 
            tl_model_uri = train_run.data.tags['tl_model_uri']
        else:
            print("\n########### Hyperparams loaded from source model ###########")
            logging.info("\n########### Hyperparams loaded from source model ###########")
            print(best_run)
            if(train_kwargs['tl_model_uri'] == 'None'): #if there is transfer learning model uri, no need to search            
                best_run = search_proper_run()
            else:
                best_run = mlflow.get_run(run_id=train_kwargs['tl_model_uri'])

            print(best_run)

            transfer_params = copy.deepcopy(best_run.data.params)

            print(f'transfer_params: {transfer_params}')

            # autolog() adds additional parameters not necessary for the enrypoint
            # so we filter those parameters
            train_params = {x: transfer_params[x] for x in transfer_params if x in (train_kwargs.keys() & transfer_params.keys())}
            train_params['dir_in'] = train_kwargs['dir_in']
            train_params['countries'] = train_kwargs['countries']
            train_params['transfer_mode'] = train_kwargs['transfer_mode']     
            
            print(train_params)
            # retrieve transfer learning model from relative source run     
            if(tl_model_uri):
                train_params['tl_model_uri'] = tl_model_uri
            else:
                train_params['tl_model_uri'] = f'runs:/{best_run.info.run_id}/model' 
                 
            print(train_params)

            train_run = _get_or_run("model", parameters=train_params) 

    return train_run, train_params

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ensemble entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ensemble_step(train_kwargs, pipeline_name, train_params):
    ensemble_run = None; ensemble_params = None
    global ensemble_model_uri

    # if(pipeline_name == 'SOURCE'): return None, None #dont run ensemble on source ####################################
    
    if 'ensemble' in train_kwargs['stages'] or 'all' in train_kwargs['stages']:
        print(f"\n=========== {pipeline_name}: Ensemble Stage =============")
        logging.info(f"\n=========== {pipeline_name}: Ensemble Stage =============")
        ensemble_params = {
                'n_estimators': train_kwargs['n_estimators'],
                'transfer_mode': train_kwargs['transfer_mode']
        }

        ensemble_params.update(train_params); del ensemble_params['tl_model_uri']
        ensemble_params['transfer_mode'] = str(Transfer.NO_TRANSFER.value)
            
        ensemble_run = _get_or_run("ensemble", parameters=ensemble_params) 

        if(pipeline_name == 'SOURCE' or train_kwargs['tl_model_uri'] != 'None'):
            ensemble_model_uri = ensemble_run.data.tags['ensemble_model_uri']

    return ensemble_run, ensemble_params

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def eval_step(train_kwargs, train_params, pipeline_name):
    eval_run = None; eval_params = None
    global ensemble_model_uri
    
    if(pipeline_name == 'SOURCE'): #dont run ensemble on source ####################################
        return eval_run, eval_params 

    if 'eval' in train_kwargs['stages'] or 'all' in train_kwargs['stages']:
        print(f"\n=========== {pipeline_name}: Eval Stage =============")
        logging.info(f"\n=========== {pipeline_name}: Eval Stage =============")

        eval_params = {
                'dir_in': train_params['dir_in'],
                'countries': train_params['countries'],
                'test_years': train_params['test_years'],
                'l_window': train_params['l_window'],
                'f_horizon': train_params['f_horizon'],
                'model_uri': ensemble_model_uri,
                'transfer_mode': train_kwargs['transfer_mode']
        }
        
        print(f'eval_params: {eval_params}')
            
        eval_run = _get_or_run("eval", parameters=eval_params) 
    
    return eval_run, eval_params

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pipeline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pipeline(train_kwargs):

    pipeline_name = 'SOURCE'
    if(train_kwargs['transfer_mode'] != str(Transfer.NO_TRANSFER.value)):
        pipeline_name = 'TARGET'

    with mlflow.start_run(run_name=pipeline_name,nested=True) as pipeline_run:
        mlflow.set_tag("stage", pipeline_name)
        # if stage is set, then it runs the according function
        
        load_run, load_data_path = load_step(train_kwargs, pipeline_name)
    
        etl_run = etl_step(train_kwargs, load_run, load_data_path, pipeline_name)

        optuna_run, optuna_params = optuna_step(train_kwargs, pipeline_name)
        
        train_run, train_params = train_step(train_kwargs, pipeline_name, optuna_run)
              
        ensemble_run, ensemble_params = ensemble_step(train_kwargs, pipeline_name, train_params)
        
        eval_run, eval_params = eval_step(train_kwargs, train_params, pipeline_name)


# ################################# Click Parameters ########################################
# Remove whitespace from your arguments
@click.command()
@click.option("--stages", type=str, default='all', help='comma seperated entry point names to execute from pipeline')
@click.option("--src_countries", type=str, default="Portugal", help='csv names from dir_in used by the source model')
@click.option("--tgt_countries", type=str, default="Portugal", help='csv names from dir_in used by the target model')
@click.option("--test_case", type=str, default="1", help='case number -check TestCase enum in model_utils and test_script.py-')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ETL Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--dir_in", type=str, default='../original_data/', help="File containing csv files used by the model")
@click.option("--local_tz", type=bool, default=False, help="flag if you want local (True) or UTC (False) timezone")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna (exclusive) Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--n_trials", type=str, default="20", help='number of trials - different tuning oh hyperparams')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option('--train_years', type=str, default='2015,2016,2017,2018,2019', help='list of years to use for training set')
@click.option('--val_years', type=str, default='2020', help='list of years to use for validation set')
@click.option('--test_years', type=str, default='2021', help='list of years to use for testing set')
@click.option("--seed", type=str, default="42", help='seed used to set random state to the model')
@click.option("--max_epochs", type=str, default="200", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1", help='range of number of layers used by the model')
@click.option("--layer_sizes", type=str, default="100", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="240", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24", help='range of forecast horizon (output layer size) used by the model')
@click.option("--l_rate", type=str, default="0.0001", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="1024", help='possible batch sizes used by the model') #16,32,
@click.option("--transfer_mode", type=str, default="0", help='indicator to use transfer learning techniques')
@click.option("--tl_model_uri", type=str, default="None", help='tl_model_uri used for accessing model used for transfer learning') 
@click.option("--num_workers", type=str, default="4", help='accelerator (cpu/gpu) processesors and threads used')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ensemble (exclusive) Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--n_estimators", type=str, default="3", help='number of estimators (models) used in ensembling')

def workflow(**kwargs):
    train_kwargs = None
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run(run_name="main") as main_run:
        mlflow.set_tag("stage", "main")
        mlflow.set_tag("test_case", TestCase(int(kwargs['test_case'])).name)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Source Domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # copy click parameters, update 'countries' element with:
        #   - 'src_countries' element's value when in source domain's pipeline
        #       > NO_TRANSFER 
        #   - 'tgt_countries' element's value when in target domain's pipeline
        #       > transfer technique (if set for transfer learning)

        exclusion_list = ['test_case','src_countries','tgt_countries']
        train_kwargs = {x: kwargs[x] for x in kwargs if x not in exclusion_list}
        train_kwargs['countries'] = kwargs['src_countries']

        # in case there is already a model, to be used, this is not needed
        train_kwargs['transfer_mode'] = str(Transfer.NO_TRANSFER.value)

        if(train_kwargs['tl_model_uri'] == 'None'):
            pipeline(train_kwargs)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Target Domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if there is transfer learning
        if(kwargs['transfer_mode'] != str(Transfer.NO_TRANSFER.value)):
            train_kwargs['dir_in'] = kwargs['dir_in'] # 'dir_in' changes in etl of previous pipeline() call and needs reset
            train_kwargs['countries'] = kwargs['tgt_countries']
            train_kwargs['transfer_mode'] = kwargs['transfer_mode']
            pipeline(train_kwargs)

if __name__ == "__main__":
    print("\n=========== Experimentation Pipeline =============")
    logging.info("\n=========== Experimentation Pipeline =============")    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    workflow()