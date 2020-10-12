from util.load_data_basic import *
from util.load_sensor_data import *
from datetime import timedelta


if __name__ == '__main__':
    # Read ground truth data
    bucket_str = 'tiles-phase1-opendataset'
    root_data_path = Path(__file__).parent.absolute().parents[1].joinpath('data', bucket_str)

    # read ground-truth data
    igtb_df = read_AllBasic(root_data_path)
    nurse_df = return_nurse_df(igtb_df)

    id_list = list(nurse_df['participant_id'])
    id_list.sort()
    for id in id_list[:]:
        print('Process participant: %s' % (id))
        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == id].Shift[0] == 'Day shift' else 'night'
        hr_df, step_df = read_fitbit_data(root_data_path, id)
        if hr_df is None or step_df is None:
            continue

        hr_df = hr_df.sort_index()

        for time_str in list(step_df.index):
            end_time_str = (pd.to_datetime(time_str) + timedelta(seconds=59)).strftime(date_time_format)[:-3]

            tmp_hr = hr_df[time_str:end_time_str]
            step_df.loc[time_str, 'heart_rate'] = np.nanmean(tmp_hr['HeartRatePPG'])

        if Path.exists(Path.joinpath(Path.resolve(root_data_path).parent, 'processed')) is False:
            Path.mkdir(Path.joinpath(Path.resolve(root_data_path).parent, 'processed'))
        if Path.exists(Path.joinpath(Path.resolve(root_data_path).parent, 'processed', 'fitbit')) is False:
            Path.mkdir(Path.joinpath(Path.resolve(root_data_path).parent, 'processed', 'fitbit'))

        step_df.to_csv(Path.joinpath(Path.resolve(root_data_path).parent, 'processed', 'fitbit', id + '.csv.gz'), compression='gzip')






