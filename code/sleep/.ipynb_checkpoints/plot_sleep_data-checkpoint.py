import sys
sys.path.insert(1, '/Users/brinkley97/Documents/development/lab-kcad/tiles-day-night/code/util/')
from load_data_basic import read_AllBasic, return_nurse_df
from load_my_path import load_gzip_csv_data
from datetime import timedelta
from scipy import stats
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_sleep(plt_df_list, title, shift='day'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3.25))
    color = ['blue', 'g']

    for i in range(len(axes.flatten())):
        print("i : ", i)
        sns.set_style("white")
        sns.distplot(np.array(plt_df_list[i]['start'].dropna()), norm_hist=True, hist=True, kde=False, bins=np.arange(0, 24, 0.5),
                     ax=axes[i], color=color[0], kde_kws={'shade': True, 'linewidth': 1, 'alpha': 0.3, 'bw_adjust': 0.5})

        sns.distplot(np.array(plt_df_list[i]['end'].dropna()), norm_hist=True, hist=True, kde=False, bins=np.arange(0, 24, 0.5),
                     ax=axes[i], color=color[1], kde_kws={'shade': True, 'linewidth': 1, 'alpha': 0.3, 'bw_adjust': 0.5})

        print('Shift %s' % (shift))
        
        print("plt_df_list[i]", plt_df_list[i]['start'])
        print()
        
        sleep_array = []
        for data in np.array(plt_df_list[i]['start']):
            if data >= 12:
                # print("DATA >=12", data)
                sleep_array.append(data - 24)
            else:
                # print("DATA", data)
                sleep_array.append(data)
        # print('sleep time %.2f, %.2f' % (np.nanmean(plt_df_list[i]['start']), np.std(plt_df_list[i]['start'])))
        # print('wake time %.2f, %.2f' % (np.nanmean(plt_df_list[i]['end']), np.std(plt_df_list[i]['end'])))
        # print('wake time %.2f, %.2f' % (np.nanmean(sleep_array), np.nanstd(sleep_array)))
        
        # print(sleep_array)
        print()
        print('sleep time %.2f, %.2f, %.2f' % (np.nanmean(plt_df_list[i]['start']), np.nanmin(plt_df_list[i]['start']), np.nanmax(plt_df_list[i]['start'])))
        # print('wake time %.2f, %.2f, %.2f' % (np.nanmean(plt_df_list[i]['end']), np.nanmin(plt_df_list[i]['end']), np.nanmax(plt_df_list[i]['end'])))
        print('wake time %.2f, %.2f, %.2f' % (np.nanmean(sleep_array), np.nanmin(sleep_array) + 24, np.nanmax(sleep_array)))
        print("\n################################################")

#         axes[i].set_xlim([0, 24])
#         axes[i].set_ylim([0, 0.9])
#         axes[i].set_xlabel('Time in a Day (hour)', fontsize=13, fontweight='bold')
#         axes[i].set_title('Day Sleep')

#         axes[i].set_xticks([0, 4, 8, 12, 16, 20, 24], minor=False)
#         for label in axes[i].get_xticklabels():
#             label.set_weight('bold')
#         for label in axes[i].get_yticklabels():
#             label.set_weight('bold')
#         axes[i].xaxis.set_tick_params(labelsize=12)
#         axes[i].yaxis.set_tick_params(labelsize=12)
#         axes[0].set_ylabel('Density', fontsize=13, fontweight='bold')
#         axes[1].set_ylabel('', fontsize=13, fontweight='bold')

#         # handles, labels = axes[i].get_legend_handles_labels()
#         axes[0].legend(labels=['Sleep Start', 'Sleep End'], prop={'size': 12, 'weight':'bold'}, loc='upper right')
#         axes[1].legend(labels=['Sleep Start', 'Sleep End'], prop={'size': 12, 'weight': 'bold'}, loc='upper right')

#     axes[0].set_title('Workdays', fontdict={'fontweight': 'bold', 'fontsize': 13})
#     axes[1].set_title('Off-days', fontdict={'fontweight': 'bold', 'fontsize': 13})

#     plt.tight_layout(rect=[0, 0.01, 1, 0.93])
#     plt.figtext(0.5, 0.95, title, ha='center', va='center', fontsize=13.5, fontweight='bold')
#     # plt.savefig(os.path.join(shift + '_sleep.png'), dpi=300)
#     plt.close()


if __name__ == '__main__':
    # Read ground truth data
    # bucket_str = 'tiles-phase1-opendataset'
    bucket_str = 'tiles_dataset'
    
    # root_data_path = Path('/Volumes/Tiles/').joinpath(bucket_str)
    root_data_path = Path(__file__).parent.absolute().parents[2].joinpath('datasets', bucket_str)

    # read ground-truth data
    # please contact the author to access: igtb_day_night.csv.gz
#     if Path(os.path.realpath(__file__)).parents[2].joinpath('igtb_day_night.csv.gz').exists() == False:
#         igtb_df = read_AllBasic(root_data_path)
#         igtb_df.to_csv(Path(os.path.realpath(__file__)).parents[2].joinpath('igtb_day_night.csv.gz'))
#     igtb_df = pd.read_csv(Path(os.path.realpath(__file__)).parents[2].joinpath('igtb_day_night.csv.gz'), index_col=0)
#     days_at_work_df = read_days_at_work(root_data_path)

#     nurse_df = return_nurse_df(igtb_df)
#     sleep_stats_df = pd.read_csv(Path.joinpath(Path.cwd(), 'sleep.csv.gz'), index_col=0)

#     id_list = list(nurse_df['participant_id'])
#     id_list.sort()

#     sleep_stats_df.to_csv(Path.joinpath(Path.cwd(), 'sleep.csv.gz'), compression='gzip')
    path_to_data = "/Users/brinkley97/Documents/development/lab-kcad/datasets/tiles_dataset/figure_3/sleep/sleep.csv.gz"
    sleep_stats_df = pd.read_csv(path_to_data)
    day_sleep_df = sleep_stats_df.loc[sleep_stats_df['shift'] == 'day']
    night_sleep_df = sleep_stats_df.loc[sleep_stats_df['shift'] == 'night']

    day_sleep_work_df = day_sleep_df.loc[day_sleep_df['work'] == 'workday']
    day_sleep_off_df = day_sleep_df.loc[day_sleep_df['work'] == 'offday']
    night_sleep_work_df = night_sleep_df.loc[night_sleep_df['work'] == 'workday']
    night_sleep_off_df = night_sleep_df.loc[night_sleep_df['work'] == 'offday']

    plt_df_list = [day_sleep_work_df, day_sleep_off_df]
    plot_sleep(plt_df_list, 'PDF of Median Sleep Start and End Time (Day Shift)', shift='day')

    plt_df_list = [night_sleep_work_df, night_sleep_off_df]
    plot_sleep(plt_df_list, 'PDF of Median Sleep Start and End Time (Night Shift)', shift='night')

    print()




