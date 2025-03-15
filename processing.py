import pandas as pd
import torch
import pickle as pkl

diatoms = pd.read_csv('DataDiatomGNN_GTstudentprojectGT/DiatomInventories_GTstudentproject.csv', sep=';')
info = pd.read_csv('DataDiatomGNN_GTstudentprojectGT/PressureStatus_GTstudentproject.csv', sep=';')
print(info.head())

# add one hot encoding to diatoms
diatoms['onehot'] = pd.Categorical(diatoms['TaxonCode']).codes

# ### Create diatoms_per_sampling_operation, which maps sampling operations to a list of one-hot encoded diatoms and their abundance.
# for each info row get all diatoms onehot number that belong to that sampling operation
sampling_operations = info['SamplingOperations_code']
diatoms_per_sampling_operation = {}
for i, sampling_operation in enumerate(sampling_operations):
    # get all rows in diatoms where diagtoms['SamplingOperations_code'] == sampling_operation
    diatoms_per_sampling_operation[sampling_operation] = diatoms[diatoms['SamplingOperations_code'] == sampling_operation][['onehot', 'Abundance_pm']]

for key in diatoms_per_sampling_operation.keys():
    try:
        diatoms_per_sampling_operation[key]['Abundance_pm'] = diatoms_per_sampling_operation[key]['Abundance_pm'].apply(lambda x: float(x.replace(',', '.')))
    except AttributeError:
        print(key, 'already processed')
        
with open('diatoms_per_sampling_operation.pkl', 'wb') as f:
    pkl.dump(diatoms_per_sampling_operation, f)


diatoms_per_sampling_operation = pkl.load(open('diatoms_per_sampling_operation.pkl', 'rb'))

print(diatoms_per_sampling_operation['S02000008_20170703'])

pressures_per_sampling_operation = pd.read_csv('DataDiatomGNN_GTstudentprojectGT/PressureStatus_GTstudentproject.csv', sep=';')
pressures_per_sampling_operation = pressures_per_sampling_operation.drop_duplicates(subset=['SamplingOperations_code'])
pressures_per_sampling_operation = pressures_per_sampling_operation.set_index('SamplingOperations_code').to_dict(orient='index')

print(pressures_per_sampling_operation['S02000008_20170703'])

sampling_op_to_tensor = {}
valid_ys = ["Nitrogencompounds_Status1Y","Nitrogencompounds_Status180D","Nitrogencompounds_Status90D","Nitrates_Status1Y","Nitrates_Status180D","Nitrates_Status90D","Phosphorouscompounds_Status1Y","Phosphorouscompounds_Status180D","Phosphorouscompounds_Status90D","Acidification_Status1Y","Acidification_Status180D","Acidification_Status90D","PAH_Status1Y","PAH_Status180D","PAH_Status90D","OrganicMatter_Status1Y","OrganicMatter_Status180D","OrganicMatter_Status90D","SuspendedMatter_Status1Y","SuspendedMatter_Status180D","SuspendedMatter_Status90D","OrganicMicropollutants_Status1Y","OrganicMicropollutants_Status180D","OrganicMicropollutants_Status90D","MineralMicropollutants_Status1Y","MineralMicropollutants_Status180D","MineralMicropollutants_Status90D"]
y_map = {"High": 0, "Good": 1, "Moderate": 2, "Poor": 3, "Bad": 4, "Unassessed": -1}
for key in diatoms_per_sampling_operation:
    scaled_onehot = torch.zeros((diatoms['onehot'].max()+1))
    one_hot = torch.zeros((diatoms['onehot'].max()+1))
    scaled_onehot[diatoms_per_sampling_operation[key]['onehot'].to_list()] = torch.tensor(diatoms_per_sampling_operation[key]['Abundance_pm'].to_list())
    one_hot[diatoms_per_sampling_operation[key]['onehot'].to_list()] = 1
    ys_list = []
    skip = False
    for y in valid_ys:
        ys_list.append(y_map[pressures_per_sampling_operation[key][y]])
    sampling_op_to_tensor[key] = (scaled_onehot, one_hot, torch.tensor(ys_list))
with open('sampling_op_to_tensor.pkl', 'wb') as f:
    pkl.dump(sampling_op_to_tensor, f)