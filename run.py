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
    return utc_time.strftime("%Y-%m-%d-%H")


def ave_per_hour(tmp, interval = 1800):
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


def load_data_from_ind(ind):
    with h5py.File('dataExportForRelease/wearableDevice/20160503_BIOCHRON_E4.hdf5', 'r') as f:

        data = {k: {} for k in ['ACC', 'IBI', 'BVP', 'EDA', 'HR', 'TEMP', 'tags']}
        data['ACC']['labels'] = ['timestamp', 'x', 'y', 'z']
        data['IBI']['labels'] = ['timestamp', 'interval']
        data['HR']['labels'] = ['timestamp', 'bpm']
        data['BVP']['labels'] = ['timestamp', 'N/A']
        data['EDA']['labels'] = ['timestamp', 'microsecond']
        data['TEMP']['labels'] = ['timestamp', 'degrees']
        data['tags']['labels'] = ['timestamp']


        # ACC data
        measure = 'ACC'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/32] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = ave_per_hour(np.array(tmp))


        # BVP data
        measure = 'BVP'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/64] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = ave_per_hour(np.array(tmp))


        # EDA data
        measure = 'EDA'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/4] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = ave_per_hour(np.array(tmp))


        # IBI data
        measure = 'IBI'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = ave_per_hour(np.array(tmp))


        # HR data
        measure = 'HR'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = ave_per_hour(np.array(tmp))



        # TEMP data
        measure = 'TEMP'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                length = f[ind][time][measure].value.shape[0]
                tmp += list(map(lambda x, y: [start_time+x/4] + list(y), range(length), f[ind][time][measure].value.tolist()))

        data[measure]['data'] = ave_per_hour(np.array(tmp))

               
        # TEMP data
        measure = 'tags'
        tmp = []
        for time in list(f[ind].keys()):
            print(time)
            if measure in f[ind][time]:
                start_time = list(f[ind][time][measure].attrs.values())[0]
                tmp += list(map(lambda x: [x[0]], f[ind][time][measure].value.tolist()))


        data[measure]['data'] = np.array(tmp)

        return data


def plot(tmp):
    plt.plot(tmp[:, -1])
    plt.show()


if __name__ == '__main__':

    # FINALLY...
    data = run.load_data_from_ind('HRV15-003')
    # WE HAVE THE DATA!

    # Look what it looks like
    for key in data:
        plot(data[key]['data'])
