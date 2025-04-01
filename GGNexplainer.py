from torch_geometric.data import DataLoader, Dataset
from copy import deepcopy
import torch
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv, global_mean_pool
from torch_geometric.explain import GNNExplainer
import matplotlib.pyplot as plt
from torch_geometric.explain import ExplainerConfig, ModelConfig
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

valid_ys = ["Nitrogencompounds_Status1Y","Nitrogencompounds_Status180D","Nitrogencompounds_Status90D","Nitrates_Status1Y","Nitrates_Status180D","Nitrates_Status90D","Phosphorouscompounds_Status1Y","Phosphorouscompounds_Status180D","Phosphorouscompounds_Status90D","Acidification_Status1Y","Acidification_Status180D","Acidification_Status90D","PAH_Status1Y","PAH_Status180D","PAH_Status90D","OrganicMatter_Status1Y","OrganicMatter_Status180D","OrganicMatter_Status90D","SuspendedMatter_Status1Y","SuspendedMatter_Status180D","SuspendedMatter_Status90D","OrganicMicropollutants_Status1Y","OrganicMicropollutants_Status180D","OrganicMicropollutants_Status90D","MineralMicropollutants_Status1Y","MineralMicropollutants_Status180D","MineralMicropollutants_Status90D"]

class GraphDataset_SplitSite(Dataset):
    def __init__(self, root, sampling_op_to_tensor, y='Nitrogencompounds_Status1Y', sites=set()):
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.sampling_op_to_tensor = deepcopy(sampling_op_to_tensor)
        self.keys = list(self.sampling_op_to_tensor.keys())
        self.y = [valid_ys.index(y)]
        temp = []
        for key in self.keys:
            if self.sampling_op_to_tensor[key][2][self.y].item() == -1:
                continue
            elif key.split('_')[0] not in sites:
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
        data = torch.load(f'{self.root}/{key}.pt')
        data.x = torch.atleast_1d(data.x)
        data.y = torch.tensor([self.sampling_op_to_tensor[key][2][self.y]], dtype=torch.float)
        data.global_node_ids = data.x
        return data, key

class GCN(torch.nn.Module):
    def __init__(self, node_features, embedding_size, hidden_channels, num_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(node_features, embedding_size)
        self.conv1 = GCNConv(embedding_size + 1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out1 = torch.nn.Linear(hidden_channels, hidden_channels // 4)
        self.out2 = torch.nn.Linear(hidden_channels // 4, num_classes)

    def forward(self, x_or_data, edge_index=None, edge_attr=None, abundancies=None, batch=None, skip_embedding=False):
        if edge_index is None:
            data = x_or_data
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            abundancies = data.abundancies
            batch = data.batch
        else:
            x = x_or_data

        if not skip_embedding and x.dim() == 1:
            x = self.embedding(x)
        x = torch.cat((x, abundancies.unsqueeze(-1)), dim=-1)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.leaky_relu(self.out1(x))
        x = self.out2(x)
        return x

# load dataset test
test_inds = np.load('test_inds.npy')
sampling_op_to_tensor = pkl.load(open('sampling_op_to_tensor.pkl', 'rb'))
sampling_ops_to_remove = ['S05029820_20150602']
for samp in sampling_ops_to_remove:
    sampling_op_to_tensor.pop(samp)
sites = list(set([key.split('_')[0] for key in sampling_op_to_tensor.keys()]))
test_sites = set([sites[i] for i in test_inds])
test_dataset = GraphDataset_SplitSite('her_lvl_1', sampling_op_to_tensor, y='Nitrates_Status1Y', sites=test_sites)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = torch.load('gcn_nitrates_model.pt')
model.eval()

# explainer conf
explainer = GNNExplainer(epochs=200, return_type='log_probs')

explainer_config = ExplainerConfig(
    explanation_type='model',
    node_mask_type=MaskType.object,
    edge_mask_type=MaskType.object
)
model_config = ModelConfig(
    mode=ModelMode.multiclass_classification,
    task_level=ModelTaskLevel.graph,
    return_type='log_probs'
)
explainer.connect(explainer_config, model_config)

all_subgraphs_importance = {}

for idx in range(len(test_dataset)):
    print(f"Traitement du graphe {idx+1}/{len(test_dataset)}")
    data, key = test_dataset[idx]
    data = data.to('cpu')
    
    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    # comput pred
    target = model(
        data.x,
        data.edge_index,
        edge_attr=data.edge_attr,
        abundancies=data.abundancies,
        batch=data.batch
    )
    target = target.argmax(dim=-1).detach()
    
    x_embedded = model.embedding(data.x).detach()
        
    # explaination
    explanation = explainer(
        model,
        x_embedded,
        data.edge_index,
        target=target,
        edge_attr=data.edge_attr,
        abundancies=data.abundancies,
        batch=data.batch,
        skip_embedding=True
    )
    
    node_mask = explanation.node_mask.squeeze()
    topk = min(5, data.num_nodes)
    topk_values, topk_indices = torch.topk(node_mask, topk)
    
    # mapping gloabl index
    topk_global_ids = data.global_node_ids[topk_indices].tolist()
    
    all_subgraphs_importance[key] = {
        'node_mask': node_mask,
        'topk_indices_local': topk_indices.tolist(),
        'topk_values': topk_values.tolist(),
        'topk_global_ids': topk_global_ids,
        'num_nodes': data.num_nodes
    }

# for key, res in all_subgraphs_importance.items():
#     print(f"Graphe {key} (Nombre de nœuds: {res['num_nodes']}):")
#     print(f"  Top nodes (indices locaux): {res['topk_indices_local']} avec scores: {res['topk_values']}")
#     print(f"  Top nodes (identifiants globaux): {res['topk_global_ids']}")

global_importance = {}

for key, res in all_subgraphs_importance.items():
    for local_idx, global_id, score in zip(res['topk_indices_local'], res['topk_global_ids'], res['topk_values']):
        if global_id not in global_importance:
            global_importance[global_id] = {'count': 0, 'score_sum': 0.0}
        global_importance[global_id]['count'] += 1
        global_importance[global_id]['score_sum'] += score

# compute mean score for each diatom
for global_id, stats in global_importance.items():
    stats['average_score'] = stats['score_sum'] / stats['count']

# sort diatoms
sorted_nodes = sorted(global_importance.items(), key=lambda x: (x[1]['count'], x[1]['average_score']), reverse=True)

taxon_names = []
with open('taxon_to_onehot.txt') as f:
    taxon_names = f.read().splitlines()

print("Ranking global des nœuds importants:")
for node, stats in sorted_nodes:
    node_name = taxon_names[node - 1]
    print(f"Diatom {node_name}: frequency {stats['count']}, average score {stats['average_score']:.3f}")

with open('evaluation_files/GNNExplainer.txt', 'w') as file:
    for node, stats in sorted_nodes:
        node_name = taxon_names[node - 1]
        file.write(f"Diatom {node_name}: frequency {stats['count']}, average score {stats['average_score']:.3f}\n")

frequencies = [stats['count'] for _, stats in sorted_nodes]
average_scores = [stats['average_score'] for _, stats in sorted_nodes]
labels = [taxon_names[node - 1] for node, _ in sorted_nodes]

plt.figure(figsize=(10, 6))
plt.scatter(frequencies, average_scores, alpha=0.7)

top_labels = 10
for i in range(top_labels):
    plt.annotate(labels[i], (frequencies[i], average_scores[i]), fontsize=9, alpha=0.8)

plt.title("Diatoms importance")
plt.xlabel("Top-k frequency appearance of the Diatom")
plt.ylabel("Importance Score")
plt.grid(True)
plt.tight_layout()
plt.savefig('evaluation_files/GNNExplainer_frequency_vs_score.png')
plt.close()

