import os
import itertools
import click
from enum import IntEnum
from model_utils import ClickParams
# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

click_params = None
all_countries = None
clustered_countries = {
    'cluster_1': 'Greece,Spain,Croatia,Italy',
    'cluster_2': 'Portugal,Ireland,Serbia,France,Belgium,Netherlands,Bulgaria,Finland,Romania,',  
    'cluster_3': 'Ukraine,Hungary,Slovenia,Slovakia,Lithuania,Austria,Germany,Poland',
    'cluster_4': 'Norway,Sweden,Denmark,Czechia,Latvia,Estonia,Switzerland'
}

# all_countries = ['Austria','Belgium','Bulgaria','Croatia','Czechia','Denmark','Estonia',
#                 'Finland','France','Germany','Greece','Hungary','Ireland','Italy',
#                 'Latvia','Lithuania','Netherland','Norway','Poland','Portugal','Romania',
#                 'Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Ukraine']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def source_target_split(countries):
    """
    This function gets a list of country names and creates two arrays 
    one for src and tgt domain, where at i-th index of those
    is src/tgt country combination
    """
    # get list of lists for all possible combinations with N-1 countries
    combinations = list(itertools.combinations(countries, len(countries)-1)) 

    # prepare combinations to be used as command argument
    source_domain = []
    for comb in combinations:
        source_domain.append(','.join(str(country) for country in comb))

    # get list of missing value of combination (target domain of combination)
    target_domain = []
    for target_countries in source_domain:
        missing_country = set(countries).difference(target_countries.split(','))
        target_domain.append(list(missing_country)[0].strip("'"))
    
    return source_domain, target_domain

# Writing a Function To Remove a Suffix in Python
def removesuffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    else:
        return text

def case_0(params):
    """
    Case 0: Source and target domain is the same one country (Naive Model)
    """
    print("=============== Case 0 ===============")    

    stages = ','.join(params.stages) if isinstance(params.stages,list) else params.stages
    for idx, country in enumerate(all_countries,1):
        # if(idx < 26): continue
        # if(idx not in [14]): continue #1,2,3,6,
        print('==========================================================================')
        print(f"~~~~~~~~~~~~~~~~ Experiment no. {idx} ~~~~~~~~~~~~~~~~~~~~")               
        print(f"Source/Target Domain: {country}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")       
        command = (
            f'mlflow run . --entry-point snaive --env-manager=local ' 
            f'-P countries={country} -P tgt_country={country} -P time_steps=168 '
            f'--experiment-name=alex_naive'
        )     
        
        try:
            print(command)
            os.system(command)
        except:
            print('##################### Something bad happened #####################')
            break
        print('==========================================================================')
        break
 
def case_1(params):
    """
    Case 1: Source and target domain is the same one country
    """
    print("=============== Case 1 ===============")    

    stages = ','.join(params.stages) if isinstance(params.stages,list) else params.stages
    for idx, country in enumerate(all_countries,1):
        # if(idx not in [6]): continue #1,2,3,6,
        print('==========================================================================')
        print(f"~~~~~~~~~~~~~~~~ Experiment no. {idx} ~~~~~~~~~~~~~~~~~~~~")               
        print(f"Source/Target Domain: {country}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")       
        command = (
            f'mlflow run . --env-manager=local -P stages={stages} '
            f'-P src_countries={country} -P transfer_mode=0 -P test_case=1 ' ## add/remove max_epochs
            # f'-P max_epochs=2 -P n_trials=2 -P n_estimators=3 -P batch_size=1024 '
            f'--experiment-name=alex_case1_new'
        )     
        
        try:
            print(command)
            os.system(command)
        except:
            print('##################### Something bad happened #####################')
            break
        print('==========================================================================')
        
    
def case_2(params):
    """
    Case 2: 
        Source domain: N-1 countries
        Target domain: 1 (remaining) country
    """
    print("=============== Case 2 ===============") 
    # create source/target list of countries
    source_domain,target_domain = source_target_split(all_countries)

    stages = ','.join(params.stages) if isinstance(params.stages,list) else params.stages
    for idx, (src, tgt) in enumerate(zip(source_domain,target_domain), 1):
        # if(idx < 12): continue
        # if(idx > 12): break
        if(idx not in [22]): continue
        print('==========================================================================')
        print(f"~~~~~~~~~~~~~~~~ Experiment no. {idx} ~~~~~~~~~~~~~~~~~~~~")               
        print(f"Source Domain: {src}")
        print(f'Target Domain: {tgt}')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   
        command = (
            f'mlflow run . --env-manager=local -P stages={stages} '
            f'-P src_countries={src} -P tgt_countries={tgt} -P test_case=2 '
            f'-P transfer_mode={params.transfer_mode} '
            # f'-P max_epochs=2 -P n_trials=2 -P n_estimators=3 -P batch_size=1024 '
            f'--experiment-name=alex_case2_new'
        )     
        
        try:
            print(command)
            # os.system(command)
        except:
            print('##################### Something bad happened #####################')
            break
        print('==========================================================================')
        break
    
def case_3(params):
    idx = 0

    for cluster in clustered_countries.values():
        countries = cluster.split(',')

        # create source/target list of countries
        source_domain,target_domain = source_target_split(countries)

        stages = ','.join(params.stages) if isinstance(params.stages,list) else params.stages

        for src, tgt in zip(source_domain,target_domain):
            idx += 1
            if(idx < 1): continue
            if(idx > 1): break    
            print('==========================================================================')
            print(f"~~~~~~~~~~~~~~~~ Experiment no. {idx} ~~~~~~~~~~~~~~~~~~~~")               
            print(f"Source Domain: {src}")
            print(f'Target Domain: {tgt}')
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")  
            command = (
                f'mlflow run . --env-manager=local -P stages={stages} '
                f'-P src_countries={src} -P tgt_countries={tgt} -P test_case=3 '
                f'-P transfer_mode={params.transfer_mode} ' 
                # f'-P max_epochs=2 -P n_trials=2 -P n_estimators=3 -P batch_size=1024 '
                f'--experiment-name=alex_case3_new'
            )     

            try:
                print(command)
                # os.system(command)
            except:
                print('##################### Something bad happened #####################')
                break
            print('==========================================================================')

def case_4(params):
    """
    Case 4: Source model contains ALL countries of the dataset
            Target domain is only a single country out of those
    """
    print("=============== Case 4 ===============")    

    stages = ','.join(params.stages) if isinstance(params.stages,list) else params.stages
    source_domain = ','.join(all_countries)

    # create global MLP model
    # no recquirement for ensemble or evaluation
    print(f"~~~~~~~~~~~~~~~~ Global MLP model ~~~~~~~~~~~~~~~~~~~~")               
    command = (
        f'mlflow run . --env-manager=local -P stages=load,etl,optuna,model ' #,optuna
        f'-P src_countries={source_domain} -P tgt_countries={source_domain} -P test_case=4 ' ## add/remove max_epochs
        f'-P transfer_mode=0 '
        # f'-P max_epochs=2 -P n_trials=2 -P n_estimators=3 -P batch_size=1024 '
        f'--experiment-name=alex_global'
    )     
    print(command)
    # os.system(command)

    # test gloval MLP for each country individually
    # when each country searchs for global MLP run, it tries to find 
    # best model in a run with no transfer learning that is in the same experiment
    # Global MLP is the only model with no transfer learning in "alex_global" experiment 
    for idx, country in enumerate(all_countries,1):
        if(idx not in [4,5,17,18,21,23]): continue

        print('==========================================================================')
        print(f"~~~~~~~~~~~~~~~~ Experiment no. {idx} ~~~~~~~~~~~~~~~~~~~~")               
        print(f"Target Domain: {country}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")       
        command = ( 
            f'mlflow run . --env-manager=local -P stages=load,etl,model,ensemble,eval '
            f'-P src_countries={source_domain} -P tgt_countries={country} -P test_case=4 '
            f'-P transfer_mode={params.transfer_mode} -P tl_model_uri=70631bd157b149f1a547fcf9cf06974d ' 
            # f'-P max_epochs=2 -P n_trials=2 -P n_estimators=3 -P batch_size=1024 '
            f'--experiment-name=alex_global'
        )     

        try:
            print(command)
            os.system(command)
        except:
            print('##################### Something bad happened #####################')
            break


        print('==========================================================================')

def case_5():
    """
    Case 5: Source model contains ALL but one countries of the dataset 
            and tries to predict from the given remainder country 
    """
    print("=============== Case 4 ===============")    

    source_domain = ','.join(all_countries)

    # test gloval MLP for each country individually
    # when each country searchs for global MLP run, it tries to find 
    # best model in a run with no transfer learning that is in the same experiment
    # Global MLP is the only model with no transfer learning in "alex_global" experiment 
    for idx, country in enumerate(all_countries,1):
        cur_source_domain = source_domain.strip(f'{country},')
        # print(f'(No. {idx}: {cur_source_domain} , {country})')
        print('==========================================================================')
        print(f"~~~~~~~~~~~~~~~~ Experiment no. {idx} ~~~~~~~~~~~~~~~~~~~~")               
        print(f"Target Country: {country}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")       
        command = (
            f'mlflow run . --env-manager=local -P stages=load,etl,optuna,model,ensemble,eval '
            f'-P src_countries={cur_source_domain} -P tgt_countries={source_domain} -P test_case=5 ' ## add/remove max_epochs
            f'-P transfer_mode=0 '
            # f'-P max_epochs=2 -P n_trials=2 -P n_estimators=3 -P batch_size=1024 '
            f'--experiment-name=alex_zeroshot'
        )     
        
        try:
            print(command)
            # os.system(command)
        except:
            print('##################### Something bad happened #####################')
            break
        
        print('==========================================================================')

#### Enum used to define test case used in run of test_script  
class TestCase(IntEnum):
    NAIVE = 0
    BENCHMARK = 1
    ALL_FOR_ONE = 2 
    CLUSTER_FOR_ONE = 3
    GLOBAL = 4
    ZEROSHOT = 5

# Remove whitespace from your arguments
@click.command()
@click.option("--case", type=str, default="1", help='test cases to use')
@click.option("--stages", type=str, default='all', help='comma seperated entry point names to execute from pipeline')
@click.option("--transfer_mode", type=str, default="1", help='indicator to use transfer learning techniques')
@click.option("--experiment_name", type=str, default="alex_trash", help='indicator to use transfer learning techniques')

def test_script(**kwargs):

    global all_countries
    all_countries = os.listdir('../preprocessed_data/') #get list of countries
    all_countries = [removesuffix(x,'.csv') for x in all_countries] # remove '.csv' suffix in each one

    click_params = ClickParams(kwargs)

    if(TestCase(click_params.case) == TestCase.NAIVE): case_0(click_params)
    if(TestCase(click_params.case) == TestCase.BENCHMARK): case_1(click_params)
    if(TestCase(click_params.case) == TestCase.ALL_FOR_ONE): case_2(click_params)
    if(TestCase(click_params.case) == TestCase.CLUSTER_FOR_ONE): case_3(click_params)
    if(TestCase(click_params.case) == TestCase.GLOBAL): case_4(click_params)
    if(TestCase(click_params.case) == TestCase.ZEROSHOT): case_5()

if __name__ == "__main__":
    test_script()