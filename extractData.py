import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
filepath = "C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\data\\100000-Patients\\"

def read_patient_data(filename, sep = "\t"):
    df = pd.read_csv(filename, sep=sep)
    return df

adm_df = read_patient_data(filepath + "AdmissionsCorePopulatedTable.txt")   ### (361760, 4)  pid, aid, admStartDate, admEndDate
adm_diag_df = read_patient_data(filepath + "AdmissionsDiagnosesCorePopulatedTable.txt")  ### (361760, 4)  pid, aid, diagCode, diagDesc
labs_df = read_patient_data(filepath + "LabsCorePopulatedTable.txt")  ### (107535387, 6) pid, aid, labName, labValue, labUnits, labDateTime,
patient_df = read_patient_data(filepath + "PatientCorePopulatedTable.txt") ### (100000, 7)  pid, gender, dob, race, MaritalStatus, lang, %belowPoverty

#merge(left_df, right_df, on=’Customer_id’, how=’inner’)
adm_diag_full_df = pd.merge(adm_df, adm_diag_df,  left_on=['PatientID','AdmissionID'], right_on = ['PatientID','AdmissionID'], how='inner')
adm_diag_full_df['diagCode'] = adm_diag_full_df['PrimaryDiagnosisCode'].str.split(".").str[0]
diseaseCode = ['E08','E09','E10','E11','M05','M06','M90']  #### E08 - E11 : Diabetes;   M05 - M06 : rheumatoid arthritis;  M90 : osteonecrosis

patient_cohort_df = adm_diag_full_df.loc[adm_diag_full_df['diagCode'].isin(diseaseCode)]  ### 62343 records

patient_list = patient_cohort_df.PatientID.unique()  ### 47609 unique patients

#### find the index date: when the disease was diagnosed  ####
def find_index_date_subset(df):
    #df = adm_diag_full_df.loc[adm_diag_full_df['PatientID'] == patientid]
    df = df.sort_values(by = ['PatientID', 'AdmissionStartDate'], ascending = [True, True])
    index_event = np.where(df['diagCode'].isin(diseaseCode))
    if index_event[0][0] >= 3:
        #adm_evnt_dt = df.iloc[index_event[0][0]]['AdmissionStartDate'].to_string(header=False, index=False)
        adm_evnt_dt = df.iloc[index_event[0][0]]['AdmissionStartDate']
        df['adm_event_dt'] = adm_evnt_dt
        #print(df)
        df[['AdmissionStartDate', 'adm_event_dt']] = df[['AdmissionStartDate', 'adm_event_dt']].apply(pd.to_datetime)  # if conversion required
        df['diff_in_days'] = (df['adm_event_dt'] - df['AdmissionStartDate']).dt.days
        df['label'] = df['diagCode'].iloc[index_event[0][0]]
        #if df[(df['diff_in_days'] <= 365) & (df['diff_in_days'] > 0)].shape[0] >= 3:
        if df[(df['diff_in_days'] > 0)].shape[0] >= 3:
            #df1 = df[(df['diff_in_days'] <= 365) & (df['diff_in_days'] > 0)]
            df1 = df[(df['diff_in_days'] > 0)]
            return df1


def get_data_statistics(df):
    total_num_of_events = df.groupby(['main_label'])['main_label'].count().reset_index(name='count') #4 total num of events
    num_unique_patients = df.groupby('main_label')['PatientID'].nunique().reset_index(name='count')   #1 num of patients
    num_unique_diagCodes = df.groupby('main_label')['PrimaryDiagnosisCode'].nunique().reset_index(name='count')  #2 num of unique codes in the cohort
    df1 = df.groupby(['main_label','PatientID'])['PrimaryDiagnosisCode'].nunique().reset_index(name='count')
    num_uniq_codes_per_person = df1.groupby('main_label')['count'].mean().reset_index(name='count') #3 num of unique codes per person
    df2 = df.groupby(['main_label','PatientID'])['AdmissionID'].max().reset_index(name='count')
    avg_num_visits = df2.groupby('main_label')['count'].mean().reset_index(name='count') #5 avg num of visits
    avg_num_events = total_num_of_events['count']/num_unique_patients['count'] #6 avg num of events per patient
    Cohort = ['# Patients', '# unique codes in cohort', '# unique codes per person', 'Total # events', 'Avg # of visits', 'Avg # event per patient']
    Diabetes = [num_unique_patients['count'].iloc[0], num_unique_diagCodes['count'].iloc[0], num_uniq_codes_per_person['count'].iloc[0], total_num_of_events['count'].iloc[0], avg_num_visits['count'].iloc[0], avg_num_events[0]]
    Osteonecrosis = [num_unique_patients['count'].iloc[1], num_unique_diagCodes['count'].iloc[1], num_uniq_codes_per_person['count'].iloc[1], total_num_of_events['count'].iloc[1], avg_num_visits['count'].iloc[1], avg_num_events[1]]
    RheumatoidArthritis = [num_unique_patients['count'].iloc[2], num_unique_diagCodes['count'].iloc[2], num_uniq_codes_per_person['count'].iloc[2], total_num_of_events['count'].iloc[2], avg_num_visits['count'].iloc[2], avg_num_events[2]]
    tab_df = pd.DataFrame({'Cohort': Cohort, 'Diabetes': Diabetes, 'Osteonecrosis': Osteonecrosis, 'Rheumatoid Arthritis': RheumatoidArthritis})
    tab_df.to_csv("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\dataStatistics.csv")
    return tab_df

final_df = pd.DataFrame()
for x, patientid in enumerate(patient_list):
    df = adm_diag_full_df.loc[adm_diag_full_df['PatientID'] == patientid]
    res_df = find_index_date_subset(df)
    final_df = final_df.append(res_df)
    final_df.to_csv("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\rawData.csv")

final_df = pd.read_csv("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\rawData.csv")

final_df['main_label'] = 'Diabetes'
final_df.loc[final_df['label'] == 'M05', 'main_label'] = 'Rheumatoid Arthritis'
final_df.loc[final_df['label'] == 'M06', 'main_label'] = 'Rheumatoid Arthritis'
final_df.loc[final_df['label'] == 'M90', 'main_label'] = 'Osteonecrosis'

tab_df = get_data_statistics(final_df)
### num of unique patients: 7862
### max of visit id : 11
### num of unique diagCodes: 522
diagCodes_list = final_df['diagCode'].unique()
diagCodes_list.sort()
patient_id_list = final_df['PatientID'].unique()
max_visits = final_df['AdmissionID'].max()


######################################################################################  For CNN model : input tensor
diagCodesFreq = final_df.groupby(['diagCode'])['PatientID'].nunique().reset_index()
diagCodesFreq = diagCodesFreq.sort_values(['PatientID'], ascending=[False])
## Take the first 28 diagCodes
diagCodes_list1 = diagCodesFreq.diagCode[0:29]
#diagCodes_list1.sort_values(ascending=True, inplace=True)
final_df2 = final_df.loc[final_df['diagCode'].isin(diagCodes_list1)]

pred_label_df = final_df2[['PatientID', 'main_label']].groupby(['PatientID', 'main_label']).count().reset_index()
pred_label_df = pred_label_df.sort_values(['PatientID'])
pred_label = pred_label_df.main_label
proc_lab_final_df = final_df2.sort_values(['PatientID','AdmissionID'])

data_df = pd.get_dummies(proc_lab_final_df[['PatientID','AdmissionID','main_label','diagCode']],columns=['diagCode'],prefix='Diag',drop_first=True)

#### group by 'PatientID','AdmissionID','main_label' and sum
####  create a matrix
data_df1 = data_df.groupby(['PatientID','AdmissionID','main_label']).sum().reset_index()
data_df1.sort_values(['PatientID', 'AdmissionID'])

patient_id_list = data_df1['PatientID'].unique()
#max_visits = data_df1['AdmissionID'].max()
max_visits = 28

ncols = 28
data_mat = np.zeros((patient_id_list.shape[0], max_visits, ncols))   # (5599, 28, 28)


def create_visit_embed_mat1(df):
    visit_event_values = df.iloc[:, -ncols:]
    visit_event_values = visit_event_values.to_numpy()
    dmat = np.zeros((max_visits - visit_event_values.shape[0], ncols))
    res_df = np.concatenate([visit_event_values, dmat])
    return res_df

final_patient_list = pred_label_df.PatientID

for ind,pid in enumerate(final_patient_list):
    print(ind)
    df = data_df1.loc[data_df1['PatientID'] == pid]
    data_mat[ind] = create_visit_embed_mat1(df)

pred_label_df['pred_label'] = 1
pred_label_df.loc[pred_label_df['main_label'] == "Diabetes", 'pred_label'] = 1
pred_label_df.loc[pred_label_df['main_label'] == "Rheumatoid Arthritis", 'pred_label'] = 2
pred_label_df.loc[pred_label_df['main_label'] == "Osteonecrosis", 'pred_label'] = 3

pre_label_arr = pred_label_df.pred_label
pred_label_arr = pre_label_arr.to_numpy()

import torch
import pickle

data_mat_mod = data_mat.reshape(data_mat.shape[0], 1, data_mat.shape[1], data_mat.shape[2])
ntrain = round(0.75*data_mat_mod.shape[0])
train_data = torch.tensor(data_mat_mod[0:ntrain].astype(int))
train_labels = torch.tensor(pred_label_arr[0:ntrain])

test_data = torch.tensor(data_mat_mod[ntrain:data_mat_mod.shape[0]].astype(int))
test_labels = torch.tensor(pred_label_arr[ntrain:data_mat_mod.shape[0]])

train_data = train_data.float()
test_data = test_data.float()
patient_dataset = {"train_data":train_data, "train_labels":train_labels, "test_data":test_data, "test_labels":test_labels}


#save
with open('C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\tensor_data.pickle', 'wb') as handle:
    pickle.dump(patient_dataset, handle)


#############################################################  For baseline models input as raw vectors

lab_filtered_df = labs_df.loc[labs_df['PatientID'].isin(patient_id_list)]
lab_filtered_df1 = lab_filtered_df.groupby(['PatientID','AdmissionID','LabName'])['LabValue'].mean().reset_index(name='AvgLabValue')
lab_filtered_df2 = lab_filtered_df1.pivot(index=['PatientID', 'AdmissionID'], columns='LabName', values='AvgLabValue').reset_index()
proc_lab_final_df = pd.merge(final_df, lab_filtered_df2,  left_on=['PatientID','AdmissionID'], right_on = ['PatientID','AdmissionID'], how='inner')


pred_label_df = proc_lab_final_df[['PatientID', 'main_label']].groupby(['PatientID', 'main_label']).count().reset_index()
pred_label_df = pred_label_df.sort_values(['PatientID'])
pred_label = pred_label_df.main_label
proc_lab_final_df = proc_lab_final_df.sort_values(['PatientID','AdmissionID'])

labValue_cols = 35  # 35 labtests/values available in the data
data_mat = np.zeros((patient_id_list.shape[0], max_visits, labValue_cols))   # (7862,11,35)

def create_visit_embed_mat(df):
    visit_event_values = df.iloc[:, -labValue_cols:]
    visit_event_values = visit_event_values.to_numpy()
    dmat = np.zeros((max_visits - visit_event_values.shape[0], labValue_cols))
    res_df = np.concatenate([visit_event_values, dmat])
    return res_df

final_patient_list = pred_label_df.PatientID

for ind,pid in enumerate(final_patient_list):
    df = proc_lab_final_df.loc[proc_lab_final_df['PatientID'] == pid]
    data_mat[ind] = create_visit_embed_mat(df)


####   saving the files   ####
data_mat = np.nan_to_num(data_mat)
np.save('C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\data_mat.npy', data_mat) # save
proc_lab_final_df.to_pickle("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\proc_lab_final_df.pkl")
pred_label_df.to_pickle("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\pred_label_df.pkl")

#data_mat_mod = np.load('C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\data_mat.npy') # load
#data_mat_mod = np.nan_to_num(data_mat_mod)

#proc_lab_final_df_mod = pd.read_pickle("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\proc_lab_final_df.pkl")
#pred_label_df_mod = pd.read_pickle("C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\pred_label_df.pkl")
