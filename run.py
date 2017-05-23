import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime


# Compress data over hours
def print_time(t):
    utc_time = datetime.utcfromtimestamp(float(t))
    print(utc_time.strftime("%Y-%m-%d %H:%M:%S"))


def timestamp_to_hour(t):
    utc_time = datetime.utcfromtimestamp(float(t))
    return utc_time.strftime("%Y-%m-%d_%H:%M:%S")


def ave_by_time(interval = 1800):
    ''' Average data according to time.'''
    def _f(tmp):

        ans = []
        i = 0

        while i < tmp.shape[0]-1:

            start_timesampe = float((int(tmp[i, 0]) // interval) * interval)
            start_hour = timestamp_to_hour(start_timesampe)
            next_timestamp = start_timesampe + interval

            for j in range(i, tmp.shape[0]):
                if tmp[j, 0] > next_timestamp or j == tmp.shape[0] - 1:
                    ans.append([start_hour] + np.mean(tmp[i:j, 1:], axis=0).tolist())
                    i = j
                    break

        return np.array(ans)

    return _f


def load_data_from_ind(ind, process = ave_by_time(), raw = False):
    '''Load Duke data and preprocess it.'''
    with h5py.File('dataExportForRelease/wearableDevice/20160503_BIOCHRON_E4.hdf5', 'r') as f:

        data = {k: {} for k in ['ACC', 'IBI', 'BVP', 'EDA', 'HR', 'TEMP', 'tags']}
        data['ACC']['labels'] = ['timestamp', 'ACC_x', 'ACC_y', 'ACC_z']
        data['IBI']['labels'] = ['timestamp', 'IBI']
        data['HR']['labels'] = ['timestamp', 'HR']
        data['BVP']['labels'] = ['timestamp', 'BVP']
        data['EDA']['labels'] = ['timestamp', 'EDA']
        data['TEMP']['labels'] = ['timestamp', 'TEMP']
        data['tags']['labels'] = ['timestamp']


        # ACC data
        measure = 'ACC'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/32] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = process(np.array(tmp)) if raw == False else np.array(tmp)


        # BVP data
        measure = 'BVP'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/64] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = process(np.array(tmp)) if raw == False else np.array(tmp)


        # EDA data
        measure = 'EDA'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/4] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = process(np.array(tmp)) if raw == False else np.array(tmp)


        # IBI data
        measure = 'IBI'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x: [start_time+x[0], x[1]], f[ind][time][measure].value.tolist()))

        data[measure]['data'] = process(np.array(tmp)) if raw == False else np.array(tmp)


        # HR data
        measure = 'HR'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = process(np.array(tmp)) if raw == False else np.array(tmp)


        # TEMP data
        measure = 'TEMP'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/4] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = process(np.array(tmp)) if raw == False else np.array(tmp)

               
        # TEMP data
        measure = 'tags'
        print('Processing ' + measure + ' data')
        tmp = []
        for time in list(f[ind].keys()):
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                tmp += list(map(lambda x: [x[0]], f[ind][time][measure].value.tolist()))

        data[measure]['data'] = np.array(tmp)

        return data


def plot(tmp):
    plt.plot(tmp[:, -1])
    plt.show()


def merge(data):
    '''Transform to Pandas dataframe.'''
    # NOTE: We throw away tags since we don't know what to do with it now
    tmp = {}
    for key in data:
        tmp[key] = pd.DataFrame(data[key]['data'], columns = data[key]['labels'])

    ans = tmp['ACC']
    for key in ['BVP', 'EDA', 'HR', 'TEMP']:
        ans = tmp[key].merge(ans, how='right', on='timestamp')

    return ans


if __name__ == '__main__':

    for ind in ['HRV15-002', 'HRV15-003', 'HRV15-004', 'HRV15-005', 'HRV15-006', 'HRV15-007', 'HRV15-008', 'HRV15-009', 'HRV15-011', 'HRV15-012', 'HRV15-013', 'HRV15-014', 'HRV15-015', 'HRV15-017', 'HRV15-018', 'HRV15-019', 'HRV15-020', 'HRV15-021', 'HRV15-022', 'HRV15-023', 'HRV15-024']:
        # FINALLY...
        data = load_data_from_ind(ind)
        # WE HAVE THE DATA!

        # NOTE: We throw away tags since we don't know what to do with it now
        data = merge(data)
        data.to_csv(ind + '.csv')

