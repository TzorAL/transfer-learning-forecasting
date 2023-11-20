import sys
import tempfile
import requests
import mlflow
import os
import click
import logging
import shutil
import pycountry
from model_utils import download_mlflow_file

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

def find_country(filename):
    # Get lists of country names and codes based on pycountry lib
    country_names = list(map(lambda country : country.name, pycountry.countries))
    country_codes = list(map(lambda country : country.alpha_2, pycountry.countries))

    cand_countries = [country for country in country_names + ["Czech Republic"] + country_codes 
                    if country in filename and not country in ["BZ", "BA"]]
    if len(cand_countries) >= 1:
        #Always choose the first candidate country
        country_name = pycountry.countries.search_fuzzy(cand_countries[0])[0].name
        country_code = pycountry.countries.search_fuzzy(cand_countries[0])[0].alpha_2
        print(f"File \"{filename}\" belongs to \"{country_name}\" with code \"{country_code}\"")
        return country_name, country_code
    else:
        return None, None

@click.command()
@click.option("--dir_in", 
              type=str, 
              default="file:///C:/Users/User/Desktop/thesis/code/pipeline/mlruns/0/f163c5e155b04a779a2615ece69da35e/artifacts/preprocessed_data",
              help='path to input, can be filepath or remote filepath: http address or artifact URI ')
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')

def load_raw_data(dir_in, countries):
    """
    series_uri: folder or file used as input for pipeline
    """
    with mlflow.start_run(run_name="load_data",nested=True) as load_start:
        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        if(not os.path.exists(dir_in)):
            input_path = download_mlflow_file(dir_in, dst_dir=None)
        else:
            input_path = dir_in

        input_path = input_path.replace('/', os.path.sep)

        if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
        with tempfile.TemporaryDirectory(dir='./temp_files/') as load_tmpdir: 
            for root, dirs, files in os.walk(input_path):
                for name in files:
                    country_name, country_code = find_country(name)
                    if(not (country_name in countries.split(',') or country_code in countries.split(','))):
                        print(f'Country \"{country_name}\" not in list for preprocessing. Ignoring...')
                        continue                    
                    print(f'Found file with name \"{name}\"')
                    shutil.copyfile(f'{input_path}/{name}', f'{load_tmpdir}/{name}')

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_param("dir_in", dir_in)
            mlflow.log_artifacts(load_tmpdir, "load_data")
            mlflow.set_tag("run_id", load_start.info.run_id)        
            mlflow.set_tag('load_data_uri', f'{load_start.info.artifact_uri}/load_data')
            mlflow.set_tag("stage", "load")

if __name__ == '__main__':
    print("\n=========== Load Data =============")
    logging.info("\n=========== Load Data =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    load_raw_data()