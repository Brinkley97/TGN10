import os, errno
import pandas as pd
import numpy as np
from pathlib import Path

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

file_type = ".csv.gz"
# Load sleep data
def read_sleep_data(data_directory, id):
    
    base = "/Users/brinkley97/Documents/development/lab-kcad/"
    path_to_file = "datasets/tiles_dataset/fitbit/sleep-metadata/"
    
    
    sleep_metadata_files = base + path_to_file + str(id) + file_type
    
    # if Path.exists(Path.joinpath(data_directory, 'fitbit', 'sleep-metadata', id + '.csv.gz')) is False:
    #     return None
    # sleep_metadata_df = pd.read_csv(Path.joinpath(data_directory, 'fitbit', 'sleep-metadata', id + '.csv.gz'))
    
    sleep_metadata_df = pd.read_csv(sleep_metadata_files)
    save_col = ['isMainSleep', 'startTime', 'endTime', 'dateOfSleep',
                'timeInBed', 'minutesAsleep', 'minutesAwake',
                'duration', 'efficiency', 'minutesAfterWakeup', 'minutesToFallAsleep']
    sleep_metadata_df = sleep_metadata_df[save_col]
    sleep_metadata_df = sleep_metadata_df.set_index('startTime')
    return sleep_metadata_df


# Load realizd data
def read_realizd_data(data_directory, id):
    
    base = "/Users/brinkley97/Documents/development/lab-kcad/"
    path_to_file = "datasets/tiles_dataset/fitbit/sleep-metadata/"
    
    
    readlizd_files = base + path_to_file + str(id) + file_type
    
    if Path.exists(Path.joinpath(data_directory, 'realizd', id + '.csv.gz')) is False:
        return None

    # realizd_df = pd.read_csv(Path.joinpath(data_directory, 'realizd', id + '.csv.gz'), index_col=1)
    realizd_df = pd.read_csv(readlizd_files, index_col=1)

    realizd_df = realizd_df.sort_index()

    time_diff_list = list(pd.to_datetime(realizd_df['session_stop']) - pd.to_datetime(realizd_df.index))
    for i in range(len(realizd_df)):
        realizd_df.loc[realizd_df.index[i], 'duration'] = time_diff_list[i].total_seconds()
    return realizd_df


# Load fitbit data
def read_fitbit_data(data_directory, id):
    
    base = "/Users/brinkley97/Documents/development/lab-kcad/"
    path_to_file = "datasets/tiles_dataset/fitbit/sleep-metadata/"
    
    
    hr_files = base + path_to_file + str(id) + file_type
    sc_files = base + path_to_file + str(id) + file_type
    
#     if Path.exists(Path.joinpath(data_directory, 'fitbit', 'heart-rate', id + '.csv.gz')) is False:
#         return None, None

#     hr_df = pd.read_csv(Path.joinpath(data_directory, 'fitbit', 'heart-rate', id + '.csv.gz'), index_col=0)
#     step_df = pd.read_csv(Path.joinpath(data_directory, 'fitbit', 'step-count', id + '.csv.gz'), index_col=0)
    hr_df = pd.read_csv(hr_files, index_col=1)
    step_df = pd.read_csv(sc_files, index_col=1)

    return hr_df, step_df


# Load fitbit data
def read_prossed_fitbit_data(data_directory, id):
    # print(data_directory, id)
    
    base = "/Users/brinkley97/Documents/development/lab-kcad/"
    path_to_file = "datasets/tiles_dataset/fitbit/sync/"
    
    file_type = ".csv.gz"
    sync_files = base + path_to_file + str(id) + file_type
    
    # if Path.exists(Path.joinpath(Path.resolve(data_directory), 'processed', 'fitbit', id + '.csv.gz')) is False:
    #     return None
    # fitbit_df = pd.read_csv(Path.joinpath(Path.resolve(data_directory), 'processed', 'fitbit', id + '.csv.gz'), index_col=0)
    fitbit_df = pd.read_csv(sync_files, index_col=0)
    return fitbit_df


# Load fitbit daily data
def read_fitbit_daily_data(data_directory, id):
    base = "/Users/brinkley97/Documents/development/lab-kcad/"
    path_to_file = "datasets/tiles_dataset/fitbit/daily-summary/"
    
    daily_summary_type = ".csv.gz"
    daily_summary_files = base + path_to_file + str(id) + daily_summary_type
    
    # if Path.exists(Path.joinpath(data_directory, 'fitbit', 'daily-summary', id + '.csv.gz')) is False:
    #     return None
    # summary_df = pd.read_csv(Path.joinpath(data_directory, 'fitbit', 'daily-summary', id + '.csv.gz'), index_col=0)
    fitbit_df = pd.read_csv(daily_summary_files, index_col=0)
        
    return summary_df