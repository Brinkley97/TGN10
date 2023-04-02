# from util.load_data_basic import *
# from util.load_sensor_data import *
import statsmodels.api as sm
from datetime import timedelta
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support

date_time_format = '%Y-%m-%dT%H:%M:%S.%f'


import sys, os
sys.path.insert(1, '/Users/brinkley97/Documents/development/lab-kcad/tiles-day-night/code/util/')
from load_data_basic import read_AllBasic, return_nurse_df, read_MGT
from load_sensor_data import read_sleep_data, read_realizd_data, read_fitbit_data, read_prossed_fitbit_data, read_fitbit_daily_data
from pathlib import Path
import pandas as pd
import numpy as np

def process_mgt(mgt_df, fitbit_df, sleep_metadata_df, maximum_hr, median_value, type='anxiety'):
    save_df = pd.DataFrame()
    mgt_df = mgt_df.dropna()
    # print(mgt_df)
    for complete_time, row_df in mgt_df.iterrows():
        # print("complete_time: ", complete_time)
        time_str = pd.to_datetime(pd.to_datetime(complete_time[:-6])).strftime(date_time_format)[:-3]
        save_row_df = pd.DataFrame(index=[time_str])
        save_row_df['id'] = id_idx
        save_row_df['shift'] = shift_str
        save_row_df['type'] = type
        save_row_df['score'] = 1 if row_df[type] > median_value else 0

        start_str = (pd.to_datetime(time_str) - timedelta(hours=6)).strftime(date_time_format)[:-3]
        start_sleep_str = (pd.to_datetime(time_str) - timedelta(hours=24)).strftime(date_time_format)[:-3]
        tmp_df = fitbit_df[start_str:time_str]
        tmp_sleep_df = sleep_metadata_df[start_sleep_str:time_str]

        if len(tmp_df) < 60 * 3 or len(tmp_sleep_df) == 0:
            continue

        save_row_df['rest'] = np.nanmean(0.5 * maximum_hr > np.array(tmp_df['heart_rate']))
        save_row_df['moderate'] = np.nanmean((0.5 * maximum_hr <= np.array(tmp_df['heart_rate'])) & (np.array(tmp_df['heart_rate']) < 0.7 * maximum_hr))
        save_row_df['vigorous'] = np.nanmean((0.7 * maximum_hr <= np.array(tmp_df['heart_rate'])) & (np.array(tmp_df['heart_rate']) < 0.85 * maximum_hr))
        save_row_df['intense'] = np.nanmean((0.85 * maximum_hr <= np.array(tmp_df['heart_rate'])))
        save_row_df['step'] = np.nanmean(tmp_df['StepCount'])

        tmp_sleep_row_df = tmp_sleep_df.iloc[-1, :]
        save_row_df['efficiency'] = tmp_sleep_row_df['efficiency']
        save_row_df['duration'] = np.abs((pd.to_datetime(tmp_sleep_df.index[-1]) - pd.to_datetime(tmp_sleep_row_df['endTime'])).total_seconds() / 60)
        save_row_df['minutesAsleep'] = tmp_sleep_row_df['minutesAsleep'] / save_row_df['duration'] * 100
        if save_row_df['minutesAsleep'][0] == 0:
            save_row_df['minutesAsleep'] = np.nan

        save_df = save_df.append(save_row_df)
        print(save_df)

    if len(save_df) < 10:
        return pd.DataFrame()

    save_df[['minutesAsleep']] = save_df[['minutesAsleep']].fillna(value=np.nanmean(save_df['minutesAsleep']))
    return save_df


if __name__ == '__main__':
    # Read ground truth data
    # bucket_str = 'tiles-phase1-opendataset'
    # root_data_path = Path(__file__).parent.absolute().parents[1].joinpath('data', bucket_str)
    
    base = "/Users/brinkley97/Documents/development/lab-kcad/"
    path_to_file = "datasets/tiles_dataset/"
    name_of_file = "table_3/igtb_day_night.csv.gz"
    igtb_file = "igtb.csv"
    # root_data_path = base + path_to_file + name_of_file
    # root_data_path = base + path_to_file + igtb_file
    root_data_path = base + path_to_file


    igtb_df = read_AllBasic(root_data_path)
#     psqi_raw_igtb = read_PSQI_Raw(file)
#     igtb_raw = read_IGTB_Raw(file)

#     nurse_df = return_nurse_df(igtb_df)
    
    # nurse_df_file = "nursedf.zip"
    # nurse_file = base + path_to_file + nurse_df_file
    # # data_df_file = base + path_to_file + nurse_df_file
    # nurse_df = pd.read_csv(nurse_file, index_col=0)
    
    nurse_df_file = "table_3/nurse_df.zip"
    nurse_file = base + path_to_file + nurse_df_file
    nurse_df = pd.read_csv(nurse_file, index_col=0)
    # print(nurse_df)


    # Read daily EMAs
    anxiety_mgt_df, stress_mgt_df, pand_mgt_df = read_MGT(root_data_path)
    anxiety_mgt_feat_df = pd.DataFrame()

    median_anxiety = np.nanmedian(anxiety_mgt_df['anxiety'])
    median_stress = np.nanmedian(stress_mgt_df['stressd'])
    median_pand_pos = np.nanmedian(pand_mgt_df['pand_PosAffect'])
    median_pand_neg = np.nanmedian(pand_mgt_df['pand_NegAffect'])

    updated_nurse_df = nurse_df.reset_index()
    # print("updated_nurse_df", updated_nurse_df)
    nurse_id = list(updated_nurse_df.loc[0:, "participant_id"])
    # print("nurse_id", nurse_id)
    nurse_id.sort()
    
    mgt_ml_df_file = "/Users/brinkley97/Documents/development/lab-kcad/tiles-day-night/code/model/mgt_ml.csv.gz" 


    if os.path.exists(mgt_ml_df_file) == False:
        print(False)
        mgt_ml_df = pd.DataFrame()
        for id_idx, id in enumerate(nurse_id):
            print(id)
            anxiety_mgt_id_df = anxiety_mgt_df.loc[anxiety_mgt_df['participant_id'] == id]
            stress_mgt_id_df = stress_mgt_df.loc[stress_mgt_df['participant_id'] == id]
            pand_mgt_id_df = pand_mgt_df.loc[anxiety_mgt_df['participant_id'] == id]
            # shift = igtb_df.loc[igtb_df['participant_id'] == id].Shift[0]
            
            participant_filter = igtb_df['participant_id'] == id
            # print(shift_bool)
            shift = igtb_df.loc[participant_filter, 'Shift'][0]
            # print(shift)
            
            shift_str = 'day' if shift == 'Day shift' else 'night'
            # age = updated_nurse_df.loc[updated_nurse_df['participant_id'] == id].age[0]
            nurse_participant_filter = updated_nurse_df['participant_id'] == id
            age = updated_nurse_df.loc[nurse_participant_filter, 'age']
            print(age)

            fitbit_df = read_prossed_fitbit_data(root_data_path, id)
            sleep_metadata_df = read_sleep_data(root_data_path, id)

            if fitbit_df is None or sleep_metadata_df is None:
                continue

            print('process %s' % (id))

            # heart rate basic
            maximum_hr = 220 - age
            # print(maximum_hr)

            fitbit_df = fitbit_df.sort_index()
            sleep_metadata_df = sleep_metadata_df.sort_index()
            sleep_metadata_df = sleep_metadata_df.loc[sleep_metadata_df['isMainSleep'] == True]
            # print(sleep_metadata_df)

            mgt_ml_df = mgt_ml_df.append(process_mgt(anxiety_mgt_id_df, fitbit_df, sleep_metadata_df, maximum_hr, median_anxiety, type='anxiety'))
            mgt_ml_df = mgt_ml_df.append(process_mgt(stress_mgt_id_df, fitbit_df, sleep_metadata_df, maximum_hr, median_stress, type='stressd'))
            mgt_ml_df = mgt_ml_df.append(process_mgt(pand_mgt_id_df, fitbit_df, sleep_metadata_df, maximum_hr, median_pand_pos, type='pand_PosAffect'))
            mgt_ml_df = mgt_ml_df.append(process_mgt(pand_mgt_id_df, fitbit_df, sleep_metadata_df, maximum_hr, median_pand_neg, type='pand_NegAffect'))
    # mgt_ml_df.to_csv(Path.cwd().joinpath('mgt_ml.csv.gz'), compression='gzip')
        # compression_opts = dict(method='zip', archive_name='mgt_ml.csv')  
        # save_mgt_ml_df = base + path_to_file + "table3/" + 'mgt_ml.csv.zip' 
        # mgt_ml_df.to_csv(save_mgt_ml_df, index=False, compression=compression_opts)
        # print("SAVED")

    else:

        mgt_ml_df_file = "/Users/brinkley97/Documents/development/lab-kcad/tiles-day-night/code/model/mgt_ml.csv.gz" 
        mgt_ml_df = pd.read_csv(mgt_ml_df_file, index_col=0)
        print(mgt_ml_df)
        print("READ")

    
    day_mgt_ml_df = mgt_ml_df.loc[mgt_ml_df['shift'] == 'day']
    night_mgt_ml_df = mgt_ml_df.loc[mgt_ml_df['shift'] == 'night']

    mgt_cols = ['anxiety', 'stressd', 'pand_PosAffect', 'pand_NegAffect']

    result_df = pd.DataFrame()
    training_list = [day_mgt_ml_df, night_mgt_ml_df]

    for training_df in training_list:
        for col in mgt_cols:
            data_ml_df = training_df.loc[training_df['type'] == col]
            unique_day_id = list(set(data_ml_df.id))
            feat_cols = ['rest', 'moderate', 'vigorous', 'step', 'minutesAsleep', 'duration']
            x = np.array(data_ml_df[feat_cols])
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = np.array(data_ml_df['score'])

            print("training %s, %.3f" % (col, np.nanmean(np.array(y==1))))
            groups = np.array(data_ml_df['id'])

            gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
            for train_idx, test_idx in gss.split(x, y, groups):

                train_data, train_lable, train_group = x[train_idx, :], y[train_idx], groups[train_idx]
                test_data, test_lable, test_group = x[test_idx, :], y[test_idx], groups[test_idx]

                gkf = list(GroupKFold(n_splits=5).split(train_data, train_lable, train_group))
                rfc = RandomForestClassifier(random_state=42)

                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': [4, 5, 6, 7, 8],
                    'criterion': ['gini', 'entropy']
                }

                CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=gkf, scoring='f1_macro')
                CV_rfc.fit(train_data, train_lable)

                test_predict = CV_rfc.best_estimator_.predict(test_data)
                precision, recall, f_score, sum = precision_recall_fscore_support(test_lable, test_predict, average='macro')

                row_df = pd.DataFrame(index=[col])
                row_df['type'] = col

                row_df['shift'] = training_df['shift'][0]
                row_df['precision'] = precision
                row_df['recall'] = recall
                row_df['f_score'] = f_score
                for j in range(len(CV_rfc.best_estimator_.feature_importances_)):
                    row_df[feat_cols[j]] = CV_rfc.best_estimator_.feature_importances_[j]

                row_df['n_estimators'] = CV_rfc.best_estimator_.n_estimators
                row_df['max_features'] = CV_rfc.best_estimator_.max_features
                row_df['max_depth'] = CV_rfc.best_estimator_.max_depth
                row_df['criterion'] = CV_rfc.best_estimator_.criterion
                result_df = result_df.append(row_df)
                print(result_df)
    compression_opts = dict(method='zip', archive_name='mgt_prediction.csv')  
    save_results_df = base + path_to_file + "table3/" + 'mgt_prediction.csv.zip' 
    result_df.to_csv(save_results_df, index=False, compression=compression_opts)
    print("SAVED")

    # result_df.to_csv(Path.cwd().joinpath('mgt_prediction.csv.gz'), compression='gzip')



