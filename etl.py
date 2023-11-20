#!/usr/bin/env python
# coding: utf-8

# File Preprocessing 

import pytz
from datetime import datetime
from datetime import timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import logging
import warnings

from darts import TimeSeries
from darts.timeseries import TimeSeries
import os
import pycountry
import dateutil

import pathlib
import click
import shutil
import tempfile
import mlflow

# get environment variables
# from dotenv import load_dotenv
# load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tmp_folder = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Αρχικά ορίζεται η συνάρτηση read_and_validate_input που διαβάζει τα αρχεία και ελέγχει αν είναι σωστά. 
Πιο συγκεκριμένα, προκύπτει Exception αν το αρχείο δεν έχει ημερομηνίες σε αύξουσα σειρά 
(που ενδεχομένως σημαίνει ότι έχει διαβαστεί με λάθος τρόπο ή λάθος τύπο ημερομηνίας) 
ή αν δεν έχει τον σωστό αριθμό ή όνομα στήλων (2 στήλες με ονόματα Date και Load στο csv)
"""

class DatesNotInOrder(Exception):
    """
    Exception raised if dates in series_csv are not sorted.
    """
    def __init__(self):
        super().__init__("Dates in series_csv are not sorted. Check date format in input csv.")

class WrongColumnNames(Exception):
    """
    Exception raised if series_csv has wrong column names.
    """
    def __init__(self, columns):
        self.message = "series_csv must have 2 columns named Date and Load."
        self.columns = columns
        super().__init__(self.message)
    def __str__(self):
        return f'Column names provided: {self.columns}. {self.message}'

def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: str = "true"):
    """
    Validates the input after read_csv is called and
    throws apropriate exception if it detects an error

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
    day_first
        Whether to read the csv assuming day comes before the month

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe from series_csv
    """
    ts = pd.read_csv(series_csv,
                        delimiter=',',
                        header=0,
                        index_col=0,
                        parse_dates=True,
                        dayfirst=day_first)
    # print(ts.index)
    ts.sort_index(ascending=True, inplace=True)
    # print(ts.index)
    # print(ts.index.sort_values())
    if not ts.index.sort_values().equals(ts.index):
        raise DatesNotInOrder()
    elif not (len(ts.columns) == 1 and 
              ts.columns[0] == 'Load' and ts.index.name == 'Date'):
        raise WrongColumnNames(list(ts.columns) + [ts.index.name])
    return ts


"""
We then call the read_csvs syntax to read the files and process them in an appropriate way. 
More specifically, some files (DE-LU MBA.csv, Ukraine BEI CTA.csv, Ukraine IPS CTA.csv) did not have their data in chronological order. 
Thus, the sort_csvs function is called in order to sort the dates in ascending order.

The files do not have a fixed name depending on the country to which each one belongs. 
Therefore, the pycountry library is used to automatically identify the country of each file.
 Thus, the name of each country and the list of its files is stored in the country_ts dictionary.

Some countries (Italy, Ukraine, Denmark, Norway, Sweden, Denmark) are divided into multiple BZs (Betting Zones). 
This results in larger variations in the data and problems regarding 
a country with several time series would influence the results of the supermodel more than it should 
(e.g. Italy with 6 time series would influence the result more than Germany with one time series). 
Therefore, in the save_results function, the csvs referring to the same country are summed. 
Finally, we have 1 csv for each country, which is stored in the tmp_folder.
"""

#Reads each file and sorts its dates in ascending order
def sort_csvs(series_csv: str = "Load_Data/2009-2019-global-load.csv",
             day_first: str = "true"):
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    ts = pd.read_csv(series_csv,
                    delimiter=',',
                    header=0,
                    index_col=0,
                    names=['Date', 'Load'],
                    usecols=[1,2],
                    parse_dates=True,
                    date_parser=date_parser,
                    dayfirst=day_first)
    csvname = os.path.basename(series_csv)
    return ts.sort_values(by=['Date'], ascending=True)

def merge_country_csvs(): # previously know 'read_csvs'
    dir_in = click.get_current_context().params["dir_in"] # make click argument accessible by function
    countries = click.get_current_context().params["countries"]
    country_names = list(map(lambda country : country.name, pycountry.countries))
    country_codes = list(map(lambda country : country.alpha_2, pycountry.countries))
    country_ts = {}
    country_file_to_name = {}
    
    for root, dirs, files in os.walk(dir_in):
        for name in files:
            # if name has country in list, then add for preprocessing
            # country_name, country_code = find_country(name)
            # if(not (country_name in countries.split(',') or country_code in countries.split(','))):
            #     print(f'Country \"{country_name}\" not in list for preprocessing. Ignoring...')
            #     continue

            if os.path.splitext(name)[1] == ".csv":
                print(f"~~~Now processing file: {name}~~~")
                logging.info(f"~~~Now processing file: {name}~~~")
                #Sorting the ts first
                ts = sort_csvs(os.path.join(root, name))
                ts = ts[~ts.index.duplicated(keep='first')]
                ts = ts.asfreq('1H')
                #Candidate countries that this file might belong to. Checking all countries and country codes.
                #"BZ" and "BA" is never a country code in this context
                #Because full country names are checked first they have priority
                cand_countries = [country for country in country_names + ["Czech Republic"] + country_codes if country in name and not country in ["BZ", "BA"]]
                if len(cand_countries) >= 1:
                    #Always choose the first candidate country
                    first = pycountry.countries.search_fuzzy(cand_countries[0])[0].name
                    print(f"File {name} belongs to {first}")
                    logging.info(f"File {name} belongs to {first}")
                    #Append the ts to the chosen country's list
                    if first in country_ts:              
                        country_ts[first].append(ts)
                    else:
                        country_ts[first] = [ts]
                    country_file_to_name[name] = pycountry.countries.search_fuzzy(cand_countries[0])[0].name
                else:
                    print(f"No match for {name}")
                    logging.info(f"No match for {name}")

    return country_ts, country_file_to_name

def save_results(country_ts, country_file_to_name):
    multiple_ts = []
    names = []
    global tmp_folder

    for country in country_ts:
        ts = country_ts[country][0]
        #Add all ts's of the same country together
        for df in country_ts[country][1:]:
            ts = ts + df        
        ts.to_csv(f"{tmp_folder}/{country}.csv")
        multiple_ts.append(ts)
        names.append(f"{tmp_folder}/{country}.csv")
        # print(names)
    return multiple_ts, names

# country_ts, country_file_to_name = merge_country_csvs()
# multiple_ts, names = save_results(country_ts, country_file_to_name)
"""
In many time series there are zero values, and also many other outliers (i.e. values much larger or much smaller than their neighbours)
 which are probably due to error and would cause problems in model training, and in MinMaxScaling. 
 For this purpose, the remove_outliers function is used to replace the aforementioned values with NaNs. 

This function, after calculating the mean and standard deviation of each month in the dataset, 
removes those values that are more than std_dev standard deviations away from this mean. 
The default value of std_dev was chosen to be 4.5 so that only values that are several standard deviations away from the mean are removed,
which are most likely to be incorrect, and keep in the dataset the values that are less standard deviations away from the mean,
which are more likely to be far from the mean simply because of an external event (e.g. weather). 
Furthermore, since these series do not show a trend, there is no need to perform a detrend.
"""
outliers_dict = {}

def remove_outliers(ts: pd.DataFrame, 
                    name: str = "Portugal", 
                    std_dev: float = 4.5, 
                    print_removed: bool = True):
    """
    Reads the input dataframe and replaces its outliers with NaNs by removing 
    values that are more than std_dev standard deviations away from their 1 month 
    mean or both. This function works with datasets that have NaN values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    name
        The name of the country to be displayed on the plots
    std_dev
        The number to be multiplied with the standard deviation of
        each 1 month  period of the dataframe. The result is then used as
        a cut-off value as described above
    print_removed
        If true it will print the removed values
        
    Returns
    -------
    pandas.DataFrame
        The original dataframe with its outliers values replaced with NaNs
    """
    #Dates with NaN values are removed from the dataframe
    ts = ts.dropna()
    #Removing all non postive values
    a = ts.loc[ts["Load"] <= 0]
    #Calculating monthly mean and standard deviation and removing values 
    #that are more than std_dev standard deviations away from the mean
    mean_per_month = ts.groupby(lambda x: x.month).mean().to_numpy()
    mean = ts.index.to_series().apply(lambda x: mean_per_month[x.month - 1])
    std_per_month = ts.groupby(lambda x: x.month).std().to_numpy()
    std = ts.index.to_series().apply(lambda x: std_per_month[x.month - 1])
    a = pd.concat([a, ts.loc[-std_dev * std + mean > ts['Load']]])
    a = pd.concat([a, ts.loc[ts['Load'] > std_dev * std + mean]])

    #Plotting Removed values and new series
    a = a.sort_values(by='Date')
    a = a[~a.index.duplicated(keep='first')]
    if print_removed: print(f"Removed outliers from {name}: {a.shape[0]} rows")

    outliers_dict[name] = (a.shape[0], ts.shape[0])

#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(ts.index, ts['Load'], color='black', label = f'Load of {name}')
#     ax.scatter(a.index, a['Load'], color='blue', label = 'Removed Outliers')
#     plt.legend()
#     plt.show()
    res = ts.drop(index=a.index)
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(res.index, res['Load'], color='black', label = f'Load of {name}')
#     plt.legend()
#     plt.show()
    return res.asfreq('1H')

def read_csvs():
    multiple_ts = []
    names = []
    result= {}
    raw_result = {}
    global tmp_folder


    for root, dirs, files in os.walk(tmp_folder):
        for name in files:
            ts = read_and_validate_input(os.path.join(root, name))
            ts = ts[~ts.index.duplicated(keep='first')]
            ts = ts.asfreq('1H')
            result[name] = remove_outliers(ts, name)
            raw_result[name] = ts

    for key, value in outliers_dict.items():
        print(key, ' : ', value[0], ' | ', value[1])
    
    print(outliers_dict)

    # import sys; sys.exit()
    return result, raw_result

# result, raw_result = read_csvs()

import holidays
from pytz import timezone
import pytz
from calendar import isleap
def isholiday(x, holiday_list):
    if x in holiday_list:
        return True
    return False


def isweekend(x):
    if x == 6 or x == 0:
        return True
    return False


def create_calendar(timeseries, timestep_minutes, holiday_list, local_timezone):
    local_tz = click.get_current_context().params["local_tz"] # make click argument accessible by function

    calendar = pd.DataFrame(
        timeseries.index.tolist(),
        columns=['datetime']
    )
        
    calendar['year'] = calendar['datetime'].apply(lambda x: x.year)
    calendar['month'] = calendar['datetime'].apply(lambda x: x.month)
    calendar['yearweek'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%V")) - 1)
    calendar['day'] = calendar['datetime'].apply(lambda x: x.day)
    calendar['hour'] = calendar['datetime'].apply(lambda x: x.hour)
    calendar['minute'] = calendar['datetime'].apply(lambda x: x.minute)
    calendar['second'] = calendar['datetime'].apply(lambda x: x.second)
    calendar['weekday'] = calendar['datetime'].apply(lambda x: x.weekday())
    calendar['monthday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%d")) - 1)
    calendar['weekend'] = calendar['weekday'].apply(lambda x: isweekend(x))
    calendar['yearday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%j")) - 1)

    if(local_tz):
        calendar['timestamp'] = calendar['datetime'].apply(
            lambda x: x.timestamp()).astype(int)
    else:
        # first convert to utc and then to timestamp
        calendar['timestamp'] = calendar['datetime'].apply(
            lambda x: local_timezone.localize(x).replace(tzinfo=pytz.utc).timestamp()).astype(int)

    calendar['holiday'] = calendar['datetime'].apply(
        lambda x: isholiday(x.date(), holiday_list))
    WNweekday = calendar['datetime'].apply(
        lambda x: x.weekday() if not isholiday(x.date(), holiday_list) else 5 if x.weekday() == 4 else 6) 
    calendar['WN'] = WNweekday + calendar['hour']/24 + calendar['minute']/(24*60)
    return calendar

"""
Finally, the imputation function for replacing NaNs with numerical values is defined, which is an implementation of: 
https://ieeexplore.ieee.org/abstract/document/7781213. This is done using the weighted average of historical data and linear interpolation. 
The weights of each method depend exponentially on the distance of the current NaN value from the closest date value according to the formulas:
                               w = e^(a * di), result = w * L + (1 - w) * H
where L is the linear interpolation and H is the historical data. Therefore, as this distance increases, 
the contribution of the linear interpolation decreases and the contribution of the historical data increases. 

The historical data for a NaN value (here the paper definition is slightly modified) is defined as 
the average of values of the same day of the week that are also less than WNcutoff (default value 1 minute) away from the NaN time,
 less than Ycutoff (default value 3 years) distance from the time of NaN and less than YDcutoff (default value 30 days) distance from the day of the time of NaN.

Holidays are considered either Saturdays (if the actual day is Friday) or Sundays (in any other case)
"""
impute_dict = {}

def impute(ts: pd.DataFrame, 
           holidays, 
           max_thhr: int = 48, 
           a: float = 0.3, 
           WNcutoff: float = 1/(24 * 60), 
           Ycutoff: int = 3, 
           YDcutoff: int = 30, 
           debug: bool = False): 
    """
    Reads the input dataframe and imputes the timeseries using a weighted average of historical data
    and simple interpolation. The weights of each method are exponentially dependent on the distance
    to the nearest non NaN value. More specifficaly, with increasing distance, the weight of
    simple interpolation decreases, and the weight of the historical data increases. If there is
    a consecutive subseries of NaNs longer than max_thhr, then it is not imputed and returned with NaN
    values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    holidays
        The holidays of the country this timeseries belongs to
    max_thhr
        If there is a consecutive subseries of NaNs longer than max_thhr, 
        then it is not imputed and returned with NaN values
    a
        The weight that shows how quickly simple interpolation's weight decreases as 
        the distacne to the nearest non NaN value increases 
    WNcutoff
        Historical data will only take into account dates that have at most WNcutoff distance 
        from the current null value's WN(Week Number)
    Ycutoff
        Historical data will only take into account dates that have at most Ycutoff distance 
        from the current null value's year
    YDcutoff
        Historical data will only take into account dates that have at most YDcutoff distance 
        from the current null value's yearday
    debug
        If true it will print helpfull intermediate results
        
    Returns
    -------
    pandas.DataFrame
        The imputed dataframe
    """
    
    #Returning calendar of the country ts belongs to
    calendar = create_calendar(ts, 60, holidays, timezone("UTC"))
    calendar.index = calendar["datetime"]
    
    #null_dates: Series with all null dates to be imputed
    null_dates = ts[ts["Load"].isnull()].index
    
    if debug:
        for date in null_dates:
            print(date)
    
    #isnull: An array which stores whether each value is null or not
    isnull = ts["Load"].isnull().values
    
    #d: List with distances to the nearest non null value
    d = [len(null_dates) for _ in range(len(null_dates))]
    
    #leave_nan: List with all the values to be left NaN because there are
    #more that max_thhr consecutive ones
    leave_nan = [False for _ in range(len(null_dates))]
    
    #Calculating the distances to the nearest non null value that is earlier in the series
    count = 1
    for i in range(len(null_dates)):
        d[i] = min(d[i], count)
        if i < len(null_dates) - 1:
            if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(hours=1):
                count += 1
            else:
                count = 1
                
    #Calculating the distances to the nearest non null value that is later in the series
    count = 1
    for i in range(len(null_dates)-1, -1, -1):
        d[i] = min(d[i], count)
        if i > 0:
            if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(hours=1):
                count += 1
            else:
                count = 1

    #We mark this subseries so that it does not get imputed
    for i in range(len(null_dates)):
        if d[i] == max_thhr // 2:
            for ii in range(max(0, i - max_thhr // 2 + 1), min(i + max_thhr // 2, len(null_dates))):
                leave_nan[ii] = True
        elif d[i] > max_thhr // 2:
            leave_nan[i] = True
                
    #This is the interpolated version of the time series
    ts_interpolatied = ts.interpolate(inplace=False)
    
    #We copy the time series so that we don't change it while iterating
    res = ts.copy()
    
    imputed = 0
    for i, null_date in enumerate(null_dates):
        if leave_nan[i]: continue
        
        imputed = imputed + 1

        #WN: Day of the week + hour/24 + minute/(24*60). Holidays are handled as
        #either Saturdays(if the real day is a Friday) or Sundays(in every other case)
        currWN = calendar.loc[null_date]['WN']
        currYN = calendar.loc[null_date]['yearday']
        currY = calendar.loc[null_date]['year']
        
        #weight of interpolated series, decreases as distance to nearest known value increases
        w = np.e ** (-a * d[i])
        
        #Historical value is calculated as the mean of values that have at most WNcutoff distance to the current null value's
        #WN, Ycutoff distance to its year, and YDcutoff distance to its yearday
        historical = ts[(~isnull) & ((calendar['WN'] - currWN < WNcutoff) & (calendar['WN'] - currWN > -WNcutoff) &\
                                    (calendar['year'] - currY < Ycutoff) & (calendar['year'] - currY > -Ycutoff) &\
                                    (((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < YDcutoff) |\
                                    ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < YDcutoff)))]["Load"]
        
        if debug: print("~~~~~~Date~~~~~~~",null_date, "~~~~~~~Dates summed~~~~~~~~~~",historical,sep="\n")
        
        historical = historical.mean()
        
        #imputed value is calculated as a wheighted average of the histrorical value and the value from intrepolation
        res.loc[null_date] = w * ts_interpolatied.loc[null_date] + (1 - w) * historical

    # merge callendar with res
    res_with_calendar = res.merge(calendar,left_index=True, right_index=True, how = 'inner')
    
    impute_dict[ts.shape[0]] = imputed

    return res_with_calendar, null_dates

def test_impute_function(country, result, max_thhr, a):
    code = compile(f"holidays.{country}()", "<string>", "eval")
    holidays_ = eval(code)
    res, null_dates = impute(result[f"{country}.csv"], holidays_, max_thhr, a, debug=False)
    
    res_ts = TimeSeries.from_dataframe(res)
    res_ts_with_na = TimeSeries.from_dataframe(result[f"{country}.csv"])
    prev = null_dates[0]
    for curr in null_dates:
        if curr > prev + pd.offsets.DateOffset(hours=1):
            _, example_imp = res_ts.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
            example_imp, _ = example_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
            _, example_no_imp = res_ts_with_na.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
            example_no_imp, _ = example_no_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
            example_imp.plot(label="Interpolation", new_plot=True)
            example_no_imp.plot(label="Original")
        prev = curr
    _, example_imp = res_ts.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
    example_imp, _ = example_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
    _, example_no_imp = res_ts_with_na.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
    example_no_imp, _ = example_no_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
    example_imp.plot(label="Interpolation", new_plot=True)
    example_no_imp.plot(label="Original")

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

from pytz import country_timezones

def utc_to_local(temp_df, country_code):
    # Get dictionary of countries and their timezones
    timezone_countries = {country: timezone 
                            for country, timezones in country_timezones.items()
                            for timezone in timezones}
    local_timezone = timezone_countries[country_code]

    # convert dates to given timezone, get timezone info
    temp_df['Start'] = temp_df['Start'].dt.tz_convert(local_timezone)

    # remove timezone information-naive, because next localize() recquires it 
    # but keep dates to local timezone
    temp_df['Start'] = temp_df['Start'].dt.tz_localize(None)

    # localize based on timezone, ignore daylight saving time, shift forward if ambiguous datetimes
    temp_df['Start'] = temp_df['Start'].dt.tz_localize(local_timezone,
                                                        ambiguous=np.array([False] * temp_df.shape[0]),
                                                        nonexistent='shift_forward')
    # crop datetime format (remove '+timezone' suffix)
    # becomes problemaric when there are multiple timezones on same country
    # for setting frequency at imputation
    temp_df['Start'] = temp_df['Start'].dt.strftime('%Y-%m-%d %H:%M:%S')

def store_with_local_time():
    dir_in = click.get_current_context().params["dir_in"] # make click argument accessible by function
    countries = click.get_current_context().params["countries"]
    country_ts = {}
    global tmp_folder

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z')
    
    # For every csv file in folder 
    for root, dirs, files in os.walk(dir_in):
        for name in files:
            # if name has country in list, then add for preprocessing
            country_name, country_code = find_country(name)
            # if(not (country_name in countries.split(',') or country_code in countries.split(','))):
            #     print(f'Country \"{country_name}\" not in list for preprocessing. Ignoring...')
            #     continue

            print("Loading dataset: "+name)
            temp_df = pd.read_csv(os.path.join(root, name),
                                parse_dates=['Start'],
                                dayfirst=True,
                                date_parser=dateparse)
            utc_to_local(temp_df, country_code) # convert 'Start' column from utc to local datetime 
            temp_df.rename(columns={"Start": "Date"}, errors="raise", inplace=True) #rename column 'Start' to 'Date'
            temp_df = temp_df[['Date','Load']] # keep only 'Date' and 'Load' columns
            temp_df.set_index('Date', inplace=True) #set 'Date' as index (neeed for 'read_and_validate...') 
            temp_df.sort_index(ascending=True, inplace=True)

            # dictionary country_ts has a key for every country and value
            # the appended dataframes of all csv files related to that country
            if country_name in country_ts: 
                country_df = country_ts[country_name]
                country_df = pd.concat([country_df,temp_df]) 
            else:
                country_ts[country_name] = temp_df.copy()

    for country in country_ts:
        country_ts[country].to_csv(f"{tmp_folder}/{country}.csv")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help="Given a CSV file (see load_raw_data), transforms it into Parquet "
    "in an mlflow artifact called 'ratings-parquet-dir'"
)
@click.option("--dir_in",
    type=str,
    default="../original_data/",
    help="Local directory where original timeseries csvs are stored"
)
@click.option("--countries",
    type=str,
    default="Portugal",
    help="csv file names to be used for preprocessing"
)
@click.option("--local_tz",
    type=bool,
    default=True,    
    help="flag if you want local (True) or UTC (False) timezone"
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def etl(**kwargs):
    
    first_trial = False

    with mlflow.start_run(run_name='etl', nested=True) as etl_start:
        mlflow.set_tag("stage", "etl")

        # set as global because they are used by most function
        global country_ts, country_file_to_name, multiple_ts, names, tmp_folder

        if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
        # Temporary directory to store merged-countries data with no outliers (not yet imputed)
        with tempfile.TemporaryDirectory(dir='./temp_files/') as tmp_folder: 
            """
            if we want our data to be based on local timezone of 
            their country of origin
            """   
            if(kwargs['local_tz']):
                store_with_local_time()
            else: 
                country_ts, country_file_to_name = merge_country_csvs()
                multiple_ts, names = save_results(country_ts, country_file_to_name)
            
            for filename in os.listdir(tmp_folder):
                f = os.path.join(tmp_folder, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    print(f)            
            result, raw_result = read_csvs() # read csv files and remove outliers

            # create temporary directory to store datasets after inputation
            with tempfile.TemporaryDirectory(dir='./temp_files/') as preprocessed_tmpdir:

                for root, dirs, files in os.walk(tmp_folder):
                    for name in files:
                        # if any(string in countries for string in ('csv_all', csv.stem)):
                        print(f'Inputing data of csv: "{name}"')
                        logging.info(f'\nInputing data of csv: "{name}"')
                        
                        country = None
                        if(kwargs['local_tz']):
                            country, country_code = find_country(name)
                        else:
                            country = pathlib.Path(name).stem #find country of csv (its name given the circumstances)
                        
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # if(os.path.exists(f"../preprocessed_data/{name}")):
                        #     first_trial = False
                        #     shutil.copyfile(f"../preprocessed_data/{name}", f"{preprocessed_tmpdir}/{name}")   
                        #     print(f'File \"{name}\" already preprocessed')                         
                        #     continue
                        
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        code = compile(f"holidays.{country}()", "<string>", "eval") #find country's holidays
                        holidays_ = eval(code)
                        res, null_dates = impute(result[f"{name}"], holidays_, 5000, 0.2, debug=False) # impute datasets and

                        global impute_dict
                        # get last entry added by impute function
                        impute_key = list(impute_dict)[-1]

                        # create new country-as-key entry in dict
                        impute_dict[country] = (impute_key, impute_dict[impute_key])

                        # remove previous entry
                        impute_dict = {key:val for key, val in impute_dict.items() if key != impute_key}


                        print(f'Stored to "{ preprocessed_tmpdir }/{name}"')
                        logging.info(f'Stored to "{preprocessed_tmpdir}/{name}"')
                        res.to_csv(f"{preprocessed_tmpdir}/{name}") #store them on seperate folder
                        
                        if not os.path.exists('../preprocessed_data/'):
                            first_trial = True 
                            os.makedirs('../preprocessed_data/')
                        
                        if(first_trial):
                            print(f'Stored to ../preprocessed_data/{name}')
                            res.to_csv(f"../preprocessed_data/{name}") #store them on seperate folder

                print(impute_dict)
                # mlflow tags
                print("\nUploading training csvs and metrics to MLflow server...")
                logging.info("\nUploading training csvs and metrics to MLflow server...")
                mlflow.log_params(kwargs)
                mlflow.log_artifacts(preprocessed_tmpdir, "preprocessed_data")
                mlflow.set_tag("run_id", etl_start.info.run_id)
                mlflow.set_tag("stage", "etl")
                mlflow.set_tag('outdir_uri', f'{etl_start.info.artifact_uri}/preprocessed_data')

if __name__ == '__main__':
    print("\n=========== ETL =============")
    logging.info("\n=========== ETL =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)    
    etl()