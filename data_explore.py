import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

ind_list = ['HRV15-002', 'HRV15-003', 'HRV15-004', 'HRV15-005', 'HRV15-006', 
        'HRV15-007', 'HRV15-008', 'HRV15-009', 'HRV15-011', 'HRV15-012',
        'HRV15-013', #'HRV15-014',
        'HRV15-017', 'HRV15-018', 'HRV15-019',
        'HRV15-020', 'HRV15-021', 'HRV15-022', 'HRV15-023', 'HRV15-024']


# Daily symtoms
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

daily_symptoms = pd.read_table('dataExportForRelease/sqlViews/vw_dailySymptoms_release.txt')
ans = []

for ind in ind_list[:5]:
    tmp = daily_symptoms[daily_symptoms.subject_id == ind]
    for day in tmp.studyDay.unique():
        ans.append(tmp[tmp.studyDay == day].max()[['subject_id', 'studyDate', 'studyDay', 'tod', 'sx_total']])

daily_symptoms = pd.DataFrame(ans)
daily_symptoms['time'] = daily_symptoms['studyDay'].apply(lambda x: day2int[x]) + daily_symptoms['tod'].apply(lambda x: tod2int[x]) / 3

daily_symptoms = daily_symptoms[['subject_id', 'time', 'sx_total']]
daily_symptoms = daily_symptoms.sort_values('time')

# Plot SX total score
f, axarr = plt.subplots(2, sharex=True)

for ind in ind_list:
    tmp = daily_symptoms[daily_symptoms.subject_id == ind]
    axarr[0].plot(tmp.time.as_matrix(), tmp.sx_total.as_matrix())

axarr[0].set_title('SX total score')
axarr[0].set_ylabel('SX total score')
axarr[0].set_xlabel('Day')
axarr[0].legend(ind_list[:5])


# Shedding
shedding2int = {
        'Positive': 1,
        'Neg.': -1,
        'WILD': 0,
        }


shedding = pd.read_table('dataExportForRelease/sqlViews/vw_shedding_release.txt')
shedding['sheddingCall'] = shedding['sheddingCall'].apply(lambda x: shedding2int[x])
ans = []
for ind in ind_list:
    tmp = shedding[shedding.subject_id == ind]
    for day in tmp.studyDay.unique():
        ans.append(tmp[tmp.studyDay == day].max()[['subject_id', 'studyDate', 'studyDay', 'sheddingCall']])

shedding = pd.DataFrame(ans)
shedding['time'] = shedding['studyDay'].apply(lambda x: day2int[x])
shedding = shedding.sort_values('time')[['subject_id', 'time', 'sheddingCall']]

# Plot shedding call

for ind in ind_list[:5]:
    tmp = shedding[shedding.subject_id == ind]
    axarr[1].plot(tmp.time.as_matrix(), tmp.sheddingCall.as_matrix())

axarr[1].set_title('Shedding')
axarr[1].set_ylabel('Shedding call')
axarr[1].set_xlabel('Day')

plt.show()


# Gene
# HRV15-2 is infected, HRV15-3 is not.
gene_1 = pd.read_csv('gene_data/HRV15-002.csv')
gene_1['time'] = gene_1.studyDay + gene_1.tod/3
gene_1 = gene_1.sort_values('time')

f, axarr = plt.subplots(2, sharex=True)

for ind in gene_1.columns[2:-1]:
    axarr[0].plot(gene_1.time.as_matrix(), gene_1[ind].as_matrix())

axarr[0].set_title('Gene expression (infected)')
axarr[0].set_ylabel('Gene')
axarr[0].set_xlabel('Day')
axarr[0].legend(gene_2.columns[2:-1], ncol=5)


y_lim = axarr[0].get_ylim()

gene_2 = pd.read_csv('gene_data/HRV15-003.csv')
gene_2['time'] = gene_2.studyDay + gene_2.tod/3
gene_2 = gene_2.sort_values('time')


for ind in gene_2.columns[2:-1]:
    axarr[1].plot(gene_2.time.as_matrix(), gene_2[ind].as_matrix())

axarr[1].set_title('Gene expression (uninfected)')
axarr[1].set_ylabel('Gene')
axarr[1].set_xlabel('Day')
axarr[1].set_ylim(y_lim)

plt.show()



# Plot sensor data
# HRV15-2 is infected, HRV15-3 is not.
data_0 = pd.read_csv('compressed_data/HRV15-002.csv')
data_1 = pd.read_csv('compressed_data/HRV15-003.csv')

data_0['time'] = data_0.timestamp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d_%H:%M:%S'))
data_1['time'] = data_1.timestamp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d_%H:%M:%S'))


f, axarr = plt.subplots(3, 2, sharex=True)

plot_feature(axarr[0][0], data_0, data_1, 'TEMP', 'Degree', ['HRV15-002', 'HRV15-003'])
plot_feature(axarr[0][1], data_0, data_1, 'HR', 'bpm', ['HRV15-002', 'HRV15-003'])

plot_feature(axarr[1][0], data_0, data_1, 'EDA')
plot_feature(axarr[1][1], data_0, data_1, 'BVP', 'mmHg')

plot_feature(axarr[2][0], data_0, data_1, 'IBI', 'second')
plot_feature(axarr[2][1], data_0, data_1, 'ACC_x')

plt.gcf().autofmt_xdate()
plt.show()


def plot_feature(ax, data_0, data_1, col, ylabel = '', legend = None):
    ax.plot(data_0.time.as_matrix(), data_0[col].as_matrix())
    ax.plot(data_1.time.as_matrix(), data_1[col].as_matrix())
    ax.set_title(col)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    if legend != None:
        ax.legend(legend)






