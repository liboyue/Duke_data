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



def merge(data):
    '''Transform to Pandas dataframe.'''
    # NOTE: We throw away tags since we don't know what to do with it now
    tmp = {}
    for key in data:
        tmp[key] = pd.DataFrame(data[key]['data'], columns = data[key]['labels'])

    ans = tmp['ACC']
    for key in ['IBI', 'BVP', 'EDA', 'HR', 'TEMP']:
        ans = tmp[key].merge(ans, how='right', on='timestamp')

    return ans


ind_list = ['HRV15-002', 'HRV15-003', 'HRV15-004', 'HRV15-005', 'HRV15-006', 
        'HRV15-007', 'HRV15-008', 'HRV15-009', 'HRV15-011', 'HRV15-012',
        'HRV15-013', 'HRV15-017', 'HRV15-018', 'HRV15-019',
        'HRV15-020', 'HRV15-021', 'HRV15-022', 'HRV15-023', 'HRV15-024']



for ind in ind_list:
    # FINALLY...
    data = load_data_from_ind(ind)
    # WE HAVE THE SENSOR DATA!

    # NOTE: We throw away tags since we don't know what to do with it now
    data = merge(data)
    data.to_csv('compressed_data/' + ind + '.csv', index = False)


start_times = []
for ind in ind_list:
    # FINALLY...
    data = pd.read_csv('compressed_data/' + ind + '.csv')
    start_times.append(data[data.index == 0].timestamp.values[0])

start_time = max(start_times)

for ind in ind_list:
    data = pd.read_csv('compressed_data/' + ind + '.csv')
    data[data.timestamp >= start_time].to_csv('data/physiolog/' + ind + '.csv', index = False)






# Shedding over time
shedding = pd.read_table('dataExportForRelease/sqlViews/vw_shedding_release.txt')

shedding2int = {
        'Neg.': -1,
        'Positive': 1,
        'WILD': 0
        }

day2int = {
        'DayMinus4':-4,
        'DayMinus3':-3,
        'DayMinus2':-2,
        'DayMinus1':-1,
        'Day0':0,
        'Day1':1,
        'Day2':2,
        'Day3':3,
        'Day4':4,
        }

tod2int = {
        'AM': 0,
        'PM1': 1,
        'PM2': 2,
        }

shedding.sheddingCall = shedding.sheddingCall.apply(lambda x: shedding2int[x])
shedding.studyDay = shedding.studyDay.apply(lambda x: day2int[x])
shedding = shedding[['studyDate', 'studyDay', 'sheddingCall']]

for ind in ind_list:
    shedding[shedding.subject_id == ind][['studyDate', 'studyDay', 'sheddingCall']].to_csv('data/shedding/' + ind + '.csv', index = False)


# Symptoms over time
daily_symptoms = pd.read_table('dataExportForRelease/sqlViews/vw_dailySymptoms_release.txt')
daily_symptoms.studyDay = daily_symptoms.studyDay.apply(lambda x: day2int[x])
daily_symptoms.tod = daily_symptoms.tod.apply(lambda x: tod2int[x])
daily_symptoms['label'] = daily_symptoms.sx_total > daily_symptoms.sx_total.mean()

for ind in ind_list:
    daily_symptoms[daily_symptoms.subject_id == ind][['studyDate', 'studyTime', 'studyDay', 'tod', 'sx_total', 'label']].to_csv('data/symptoms/' + ind + '.csv', index = False)

time.loc[(time.subject_id == row.subject_id) & (time.studyDay == row.studyDay) & (time.tod == row.tod)]


# Shedding over time
shedding['studyDate'] = 0
shedding['studyTime'] = 0

for index, row in shedding.iterrows():
    shedding.set_value(index, 'studyDate', 100)
    row['studyDate'] = 10
    time[time.subject_id == row.subject_id]
    row['studyTime']

# Time and data mapping
time = daily_symptoms[['subject_id', 'studyDate', 'studyTime', 'studyDay', 'tod']]


# process gene data
day2int = {
        'DayMinus4':-4,
        'DayMinus3':-3,
        'DayMinus2':-2,
        'DayMinus1':-1,
        'Day0':0,
        'Day1':1,
        'Day2':2,
        'Day3':3,
        'Day4':4,
        }

tod2int = {
        'AM': 0,
        'PM1': 1,
        'PM2': 2,
        }


gene_list = ['RSAD2', 'IFI44L', 'LAMP3', 'SERPING1', 'IFI44', 'IFIT1', 'IFI44', 'ISG15', 'SIGLEC1', 'OAS3', 'HERC5', 'LOC727996', 'IFIT3', 'IFI6', 'OASL', 'IFI27', 'ATF3', 'MX1', 'OAS1', 'OAS1', 'LOC26010', 'XAF1', 'OAS', 'IFIT2', 'OAS2', 'LY6E', '210657_s_at', 'DDX58', 'TNFAIP6', 'RTP4']


gene = pd.read_table('dataExportForRelease/sqlViews/vw_mRNASeqTotalCountsFtDetrick_release.txt')
gene = gene[gene.FirstColumn.isin(gene_list)]
keys = pd.read_table('dataExportForRelease/sqlViews/vw_mRNASeqKeyFtDetrick_release.txt')


#gene[['FirstColumn'] + list(t[1]['sample_id'])]
cols = ['studyDay', 'tod'] + list(gene['FirstColumn'])


# Reorganize gene data
ans = {}
for k, v in keys.groupby(keys.subject_id):
    #print(t)
    data = []
    for _, x in v[['sample_id', 'studyDay', 'tod']].iterrows():
        if x['studyDay'] != 'FollowUp':
            data.append([day2int[x['studyDay']], tod2int[x['tod']]] + list(gene[x['sample_id']]))
    ans[k] = pd.DataFrame(data=data, columns=cols)


# Add empty rows for missing data
def add_empty_rows(data):
    data['time'] = data.studyDay * 3 + data.tod
    data = data.sort_values('time')
    times = data.time.values.tolist()
    append_list = []
    for i in range(len(times)-1):
        if times[i+1] - times[i] > 1:
            for j in range(times[i]+1, times[i+1]):
                tod = int(j % 3)
                day = int((j - tod) / 3)
                append_list.append(pd.Series([day, tod] + [np.NaN]*(len(gene.columns)-3) + [j], index=data.columns))
    for x in append_list:
        data = data.append(x, ignore_index=True)
    data = data.sort_values('time')
    del data['time']
    return data

for ind in ind_list:
    add_empty_rows(ans[ind]).to_csv('data/gene/' + ind + '.csv', index = False)

