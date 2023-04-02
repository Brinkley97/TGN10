# from util.load_data_basic import *
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys, os

sys.path.insert(1, '/Users/brinkley97/Documents/development/lab-kcad/tiles-day-night/code/util/')
from load_data_basic import read_AllBasic, return_nurse_df, read_MGT

from pathlib import Path
import pandas as pd


def read_participant_mgt(id, anxiety_mgt_df, stress_mgt_df, pand_mgt_df, work_timeline_df, shift='day', gender='Male', age=''):
    ema_anxiety_df = anxiety_mgt_df.loc[anxiety_mgt_df['participant_id'] == id]
    ema_stress_df = stress_mgt_df.loc[stress_mgt_df['participant_id'] == id]
    ema_pand_df = pand_mgt_df.loc[pand_mgt_df['participant_id'] == id]

    ema_anxiety_df, ema_stress_df, ema_pand_df = ema_anxiety_df.dropna(), ema_stress_df.dropna(), ema_pand_df.dropna()
    ema_anxiety_df, ema_stress_df, ema_pand_df = ema_anxiety_df.sort_index(), ema_stress_df.sort_index(), ema_pand_df.sort_index()

    save_df = pd.DataFrame()
    for index_work, row_df in work_timeline_df.iterrows():
        day_anxiety_df = ema_anxiety_df[row_df['start']:row_df['end']]

        if len(day_anxiety_df) == 0:
            continue
        for index_ema, row_ema_df in day_anxiety_df.iterrows():
            save_row_df = pd.DataFrame(index=[index_ema])
            save_row_df['id'] = row_ema_df['participant_id']
            save_row_df['shift'] = shift
            save_row_df['gender'] = gender
            save_row_df['age'] = age
            save_row_df['anxiety'] = row_ema_df['anxiety']
            if index_ema in list(ema_stress_df.index):
                save_row_df['stressd'] = ema_stress_df.loc[index_ema, 'stressd']
            if index_ema in list(ema_pand_df.index):
                save_row_df['pand_PosAffect'] = ema_pand_df.loc[index_ema, 'pand_PosAffect']
                save_row_df['pand_NegAffect'] = ema_pand_df.loc[index_ema, 'pand_NegAffect']

            save_row_df['work'] = row_df['work']
            save_df = save_df.append(save_row_df)

    if len(save_df) <= 10:
        return pd.DataFrame()

    return_df = pd.DataFrame()
    if len(save_df.loc[save_df['work'] == 1]) > 0:
        tmp_df = save_df.loc[save_df['work'] == 1]
        tmp_df.loc[:, 'work'] = 'work'
        return_df = pd.concat([return_df, tmp_df])

    if len(save_df.loc[save_df['work'] == 0]) > 0:
        tmp_df = save_df.loc[save_df['work'] == 0]
        tmp_df.loc[:, 'work'] = 'off'
        return_df = pd.concat([return_df, tmp_df])

    return return_df


def print_stats(day_data, night_data, col, func=stats.kruskal, func_name='K-S'):
    print(col)
    print('Number of valid participant: workday: %i; offday: %i\n' % (len(day_data), len(night_data)))

    # Print
    print('Workday: median = %.2f, mean = %.2f' % (np.nanmedian(day_data), np.nanmean(day_data)))
    print('Off-day: median = %.2f, mean = %.2f' % (np.nanmedian(night_data), np.nanmean(night_data)))

    # stats test
    stat, p = func(day_data, night_data)
    print(func_name + ' test for %s' % col)
    print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

    return p


if __name__ == '__main__':
    # Read ground truth data
#     bucket_str = 'tiles-phase1-opendataset'
#     root_data_path = Path('/Volumes/Tiles/').joinpath(bucket_str)

#     # Read demographics, igtb, etc.
    # please contact the author to access: igtb_day_night.csv.gz
#     if Path(os.path.realpath(__file__)).parents[1].joinpath('igtb_day_night.csv.gz').exists() == False:
#         igtb_df = read_AllBasic(root_data_path)
#         igtb_df.to_csv(Path(os.path.realpath(__file__)).parents[1].joinpath('igtb_day_night.csv.gz'))
#     igtb_df = pd.read_csv(Path(os.path.realpath(__file__)).parents[1].joinpath('igtb_day_night.csv.gz'), index_col=0)
    
        # Read ground truth data
    bucket_str = 'tiles_dataset'
    
    # root_data_path = Path('/Volumes/Tiles/').joinpath(bucket_str)
    root_data_path = Path(__file__).parent.absolute().parents[2].joinpath('datasets', bucket_str)
    
    # please contact the author to access: igtb_day_night.csv.gz
    igtb_day_night_file = 'igtb_day_night.csv.gz'
    # path_to_file = '/Users/brinkley97/Documents/development/lab-kcad/datasets/tiles_dataset/surveys/raw/IGTB/USC_PILOT_IGTB.csv'
    path_to_file_igtb_day_night_file = '/Users/brinkley97/Documents/development/lab-kcad/tiles-day-night/code/' + igtb_day_night_file

    # if Path(os.path.realpath(__file__)).parents[2].joinpath(path_to_file).exists() == False:
    #     igtb_df = read_AllBasic(root_data_path)
    #     igtb_df.to_csv('/Users/brinkley97/Documents/development/lab-kcad/datasets/tiles_dataset/table_1/anova/igtb_day_night.csv.gz')
    igtb_df = pd.read_csv(path_to_file_igtb_day_night_file, index_col=0)
    
    
    nurse_df = return_nurse_df(igtb_df)
    nurse_id = list(nurse_df.participant_id)
    nurse_id.sort()

    # Read daily EMAs
    anxiety_mgt_df, stress_mgt_df, pand_mgt_df = read_MGT(root_data_path)

    if Path.exists(Path.cwd().joinpath('mgt_lm1.csv.gz')) is False:
        mgt_df = pd.DataFrame()
        for id in nurse_id:
            if Path.exists(Path.cwd().joinpath('..', '..', '..', 'data', 'processed', 'timeline', id + '.csv.gz')) is False:
                continue

            gender = igtb_df.loc[igtb_df['participant_id'] == id].gender[0]
            age = igtb_df.loc[igtb_df['participant_id'] == id].age[0]
            educ = igtb_df.loc[igtb_df['participant_id'] == id].educ[0]
            lang = igtb_df.loc[igtb_df['participant_id'] == id].lang[0]

            shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == id]['Shift'].values[0] == 'Day shift' else 'night'
            gender_str = 'Male' if gender == 1 else 'Female'
            lang_str = 'Yes' if lang == 1 else 'No'
            age_str = '< 40 Years' if age < 40 else '>= 40 Years'

            timeline_df = pd.read_csv(Path.cwd().joinpath('..', '..', '..', 'data', 'processed', 'timeline', id + '.csv.gz'), index_col=0)
            mgt_df = mgt_df.append(read_participant_mgt(id, anxiety_mgt_df, stress_mgt_df, pand_mgt_df, timeline_df, shift=shift, gender=gender_str, age=age_str))
        mgt_df = mgt_df.dropna()
        mgt_df.to_csv(Path.cwd().joinpath('mgt_lm.csv.gz'), compression='gzip')
    else:
        mgt_df = pd.read_csv(Path.cwd().joinpath('mgt_lm.csv.gz'), index_col=0)

#     mgt_df = mgt_df.dropna()
#     md = smf.mixedlm("pand_NegAffect ~ shift*work", mgt_df, groups=mgt_df["id"])
#     mdf = md.fit()
#     print(mdf.summary())

#     md = smf.mixedlm("stressd ~ shift*work", mgt_df, groups=mgt_df["id"])
#     mdf = md.fit()
#     print(mdf.summary())

#     md = smf.mixedlm("anxiety ~ shift*work", mgt_df, groups=mgt_df["id"])
#     mdf = md.fit()
#     print(mdf.summary())
