# transfer-learning-forecasting

The repository for load forecasting through Transfer Learning techniques as part of the paper [Transfer Learning for Day-Ahead Load Forecasting: A Case Study on European National Electricity Demand Time Series](https://www.mdpi.com/2227-7390/12/1/19).

## Installation

This project is implemented in [MLFlow](https://mlflow.org/docs/latest/index.html) to handle the different stages in the pipeline. Each stage can be run independently as an entry point, and its inputs and outputs are stored in its respected MLflow run file -- see **MLProject** file for details regarding the inputs of each entry point.

| Entrypoint |            Filename           |
|:----------:|:-----------------------------:|
| main       | main.py                       |
| load       | load_raw_data.py              |
| etl        | etl.py                        |
| optuna     | forecasting_model_optuna.py   |
| model      | forecasting_model.py          |
| ensemble   | forecasting_model_ensemble.py |
| eval       | forecasting_model_eval.py     |
| snaive*     | forecasting_model_naive.py   |

***snaive** is an independent entrypoint used for comparative evaluation of our scenarios with naive forecasts

**model_utils.py** is a python file containing general-purpose functions used in more than one pipeline stages

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MLFlow.
```bash
pip install mlflow
```
By executing an entrypoint, **MLProject** checks package dependencies (see **python_env.yaml**) and proceeds to install them.

## Data format

Our pipeline is capable of processing multiple files from different or same countries. 
Data must: 
* all be in a single directory provided at user input (dir_in) 
* be in csv format 
* contain one datetime column (named "Start") and one column containing energy load data (named "Load")
* be in 1-hour interval
* be named after the country name (e.g Greece) or code (e.g GR) they represent

## Usage

As mentioned, **MLProject** offers a wide range of parameters that should be tuned, with regards to each stage in pipeline. 
While each entrypoint has its own parameters, **main** entrypoint contains all parameters required for any entrypoint, and distributes them accordingly:  

|   Parameters   | Type |        Default Value       |                           Description                           |
|:--------------:|:----:|:--------------------------:|:---------------------------------------------------------------:|
|     stages     |  str |            'all            |    comma seperated entry point names to execute from pipeline   |
|     stages     |  str |            'all'           |      comma-seperated containing entrypoint names to be run      |
|     dir_in     |  str |     '../original_data/'    |        Folder path containing csv files used by the model       |
|    local_tz    | bool |            False           |      flag if you want local (True) or UTC (False) timezone      |
|  src_countries |  str |         'Portugal'         |          csv names from dir_in used by the source model         |
|  tgt_countries |  str |         'Portugal'         |          csv names from dir_in used by the target model         |
|      seed      |  str |            '42'            |            seed used to set random state to the model           |
|   train_years  |  str | '2015,2016,2017,2018,2019' |              list of years to use for training set              |
|    val_years   |  str |           '2020'           |             list of years to use for validation set             |
|   test_years   |  str |           '2021'           |               list of years to use for testing set              |
|    n_trials    |  str |             '2'            |        number of trials - different tuning oh hyperparams       |
|   max_epochs   |  str |             '3'            |           range of number of epochs used by the model           |
|    n_layers    |  str |             '1'            |           range of number of layers used by the model           |
|   layer_sizes  |  str |            "100"           |          range of size of each layer used by the model          |
|    l_window    |  str |            '240'           |  range of lookback window (input layer size) used by the model  |
|    f_horizon   |  str |             '24'           | range of forecast horizon (output layer size) used by the model |
|     l_rate     |  str |          '0.0001'          |             range of learning rate used by the model            |
|   activation   |  str |           'ReLU'           |        activation functions experimented on by the model        |
| optimizer_name |  str |           'Adam'           |             optimizers experimented on by the model             |
|   batch_size   |  str |           '1024'           |             batch sizes experimented on by the model            |
|  transfer_mode |  str |             "0"            |          indicator to use transfer learning techniques          |
|   num_workers  |  str |             '2'            |       accelerator (cpu/gpu) processesors and threads used       |
|  tl_model_uri  |  str |            None            |     uri path for accessing model used for transfer learning     |
|  n_estimators  |  str |             '3'            |         number of estimators (models) used in ensembling        |
|    test_case   |  str |             '1'            |             indicator of scenario that is being used            |

**Example:** locally train a model in the Greek-Spanish dataset, apply AbO warm-start transfer learning on Italy and store in an experiment with name "full_pipeline":

```bash
mlflow run . --env-manager=local -P stages='all'
             -P src_countries='Greece,Spain' -P tgt_countries='Italy' 
             -P test_case=2 -P transfer_mode=1
             --experiment-name=full_pipeline
```

**transfer_mode** and **test_case** are integers determined by the following Enums:

```python
class TestCase(IntEnum):
    NAIVE = 0
    BASELINE = 1
    AbO = 2 
    CbO = 3

class Transfer(IntEnum):
    NO_TRANSFER = 0
    WARM_START = 1 

```

 The execution of a single entrypoint can be done using the "-e" flag.
 **Example:** execute "optuna" entrypoint and store run in an experiment with name "optuna_entrypoint": 
 ```python
 mlflow run . --env-manager=local -e optuna --experiment-name=optuna_entrypoint
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

