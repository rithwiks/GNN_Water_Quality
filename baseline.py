import pandas as pd
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_data():
    diatoms = pd.read_csv('DataDiatomGNN_GTstudentprojectGT/DiatomInventories_GTstudentproject.csv', sep=';')
    info = pd.read_csv('DataDiatomGNN_GTstudentprojectGT/PressureStatus_GTstudentproject.csv', sep=';')
    print(info.head())

    with open('taxon_to_onehot.txt', 'rb') as f:
        taxons = f.readlines()
        taxons = [x.decode('utf-8').strip() for x in taxons]
    diatoms['onehot'] = diatoms['TaxonCode'].apply(lambda x: taxons.index(x))

    diatoms_per_sampling_operation = pkl.load(open('diatoms_per_sampling_operation.pkl', 'rb'))

    print(diatoms_per_sampling_operation['S02000008_20170703'])

    pressures_per_sampling_operation = pd.read_csv('DataDiatomGNN_GTstudentprojectGT/PressureStatus_GTstudentproject.csv', sep=';')
    pressures_per_sampling_operation = pressures_per_sampling_operation.drop_duplicates(subset=['SamplingOperations_code'])
    # make pressuers_per_sampling_operation a dictionary with sampling operation as key
    pressures_per_sampling_operation = pressures_per_sampling_operation.set_index('SamplingOperations_code').to_dict(orient='index')

    print(pressures_per_sampling_operation['S02000008_20170703'])

    return diatoms, diatoms_per_sampling_operation, pressures_per_sampling_operation

def prepare_tensors(diatoms, diatoms_per_sampling_operation, pressures_per_sampling_operation):

    valid_ys = ["Nitrogencompounds_Status1Y","Nitrogencompounds_Status180D","Nitrogencompounds_Status90D","Nitrates_Status1Y","Nitrates_Status180D","Nitrates_Status90D","Phosphorouscompounds_Status1Y","Phosphorouscompounds_Status180D","Phosphorouscompounds_Status90D","Acidification_Status1Y","Acidification_Status180D","Acidification_Status90D","PAH_Status1Y","PAH_Status180D","PAH_Status90D","OrganicMatter_Status1Y","OrganicMatter_Status180D","OrganicMatter_Status90D","SuspendedMatter_Status1Y","SuspendedMatter_Status180D","SuspendedMatter_Status90D","OrganicMicropollutants_Status1Y","OrganicMicropollutants_Status180D","OrganicMicropollutants_Status90D","MineralMicropollutants_Status1Y","MineralMicropollutants_Status180D","MineralMicropollutants_Status90D"]

    sampling_op_to_tensor = {}
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
    
    return sampling_op_to_tensor

class DiatomDataset(Dataset):
    def __init__(self, sampling_op_to_tensor, x='scaled_onehot', y='Nitrogencompounds_Status1Y', valid_ys = ["Nitrogencompounds_Status1Y","Nitrogencompounds_Status180D","Nitrogencompounds_Status90D","Nitrates_Status1Y","Nitrates_Status180D","Nitrates_Status90D","Phosphorouscompounds_Status1Y","Phosphorouscompounds_Status180D","Phosphorouscompounds_Status90D","Acidification_Status1Y","Acidification_Status180D","Acidification_Status90D","PAH_Status1Y","PAH_Status180D","PAH_Status90D","OrganicMatter_Status1Y","OrganicMatter_Status180D","OrganicMatter_Status90D","SuspendedMatter_Status1Y","SuspendedMatter_Status180D","SuspendedMatter_Status90D","OrganicMicropollutants_Status1Y","OrganicMicropollutants_Status180D","OrganicMicropollutants_Status90D","MineralMicropollutants_Status1Y","MineralMicropollutants_Status180D","MineralMicropollutants_Status90D"]):
        self.sampling_op_to_tensor = sampling_op_to_tensor.copy()
        self.keys = list(self.sampling_op_to_tensor.keys())
        self.x = 0 if x == 'scaled_onehot' else 1
        self.y = [valid_ys.index(y)]
        temp = []
        for key in self.keys:
            if self.sampling_op_to_tensor[key][2][self.y].item() == -1:
                continue
            else:
                temp.append(key)
        print('Skipped', len(self.keys) - len(temp), 'samples')
        self.keys = temp

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.sampling_op_to_tensor[key][self.x], self.sampling_op_to_tensor[key][2][self.y].item()

# dataset = DiatomDataset(sampling_op_to_tensor, x='scaled_onehot', y='Nitrates_Status1Y')
# y_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
# for i in range(len(dataset)):
#     y_dist[dataset[i][1]] += 1
# print('Y distribution (0 is High, 4 is Bad):', y_dist)

class DiatomDatasetBinary(Dataset):
    def __init__(self, sampling_op_to_tensor, x='scaled_onehot', y='Nitrogencompounds_Status1Y', valid_ys = ["Nitrogencompounds_Status1Y","Nitrogencompounds_Status180D","Nitrogencompounds_Status90D","Nitrates_Status1Y","Nitrates_Status180D","Nitrates_Status90D","Phosphorouscompounds_Status1Y","Phosphorouscompounds_Status180D","Phosphorouscompounds_Status90D","Acidification_Status1Y","Acidification_Status180D","Acidification_Status90D","PAH_Status1Y","PAH_Status180D","PAH_Status90D","OrganicMatter_Status1Y","OrganicMatter_Status180D","OrganicMatter_Status90D","SuspendedMatter_Status1Y","SuspendedMatter_Status180D","SuspendedMatter_Status90D","OrganicMicropollutants_Status1Y","OrganicMicropollutants_Status180D","OrganicMicropollutants_Status90D","MineralMicropollutants_Status1Y","MineralMicropollutants_Status180D","MineralMicropollutants_Status90D"]):
        self.sampling_op_to_tensor = sampling_op_to_tensor.copy()
        self.keys = list(self.sampling_op_to_tensor.keys())
        self.x = 0 if x == 'scaled_onehot' else 1
        self.y = [valid_ys.index(y)]
        temp = []
        for key in self.keys:
            if self.sampling_op_to_tensor[key][2][self.y].item() == -1:
                continue
            else:
                temp.append(key)
                self.sampling_op_to_tensor[key][2][self.y] = 1 if self.sampling_op_to_tensor[key][2][self.y].item() >= 2 else 0
        print('Skipped', len(self.keys) - len(temp), 'samples')
        self.keys = temp

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.sampling_op_to_tensor[key][self.x], self.sampling_op_to_tensor[key][2][self.y].item(), key

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)         
        return x


def train_model(model, train_dataloader, criterion, optimizer, epochs=4):
    for epoch in range(epochs):
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(train_dataloader)):
            x, y, keys = data
            optimizer.zero_grad()
            output = model(x.float())
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if i == len(train_dataloader)-1:
                print(f'Epoch {epoch}, Loss {loss.item()}, Accuracy {100 * correct / total}, {i}/{len(train_dataloader)}')
    
    print('Finished Training')
    torch.save(model, 'diatom_model_complete.pth')
    print('Trained model saved.')


def main():
    diatoms, diatoms_per_sampling_operation, pressures_per_sampling_operation = load_data()
    sampling_op_to_tensor = prepare_tensors(diatoms, diatoms_per_sampling_operation, pressures_per_sampling_operation)
    
    input_dim = diatoms['onehot'].max()+1
    output_dim = 2

    model = Net(input_dim, output_dim, 4096, 1024, 256)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset_bin = DiatomDatasetBinary(sampling_op_to_tensor, x='scaled_onehot', y='Nitrates_Status1Y')
    y_dist_bin = {0: 0, 1: 0}
    for i in range(len(dataset_bin)):
        y_dist_bin[dataset_bin[i][1]] += 1
    print('Y distribution (0 is Good, 1 is Mediocre/Bad):', y_dist_bin)

    # split the dataset into training and test
    train_size = int(0.8 * len(dataset_bin))
    test_size = len(dataset_bin) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_bin, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
      
    train_model(model, train_dataloader, criterion, optimizer)

if __name__ == "__main__":
    main()
