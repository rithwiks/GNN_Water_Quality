import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fpdf import FPDF
from tqdm import tqdm
import pickle as pkl
from torch_geometric.data import DataLoader, Dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from copy import deepcopy



class GCN(torch.nn.Module):
    def __init__(self, node_features, embedding_size, hidden_channels, num_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(node_features, embedding_size)
        self.conv1 = GCNConv(embedding_size+1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv1 = GraphConv(embedding_size+1, hidden_channels)
        # self.conv2 = GraphConv(hidden_channels, hidden_channels)
        # self.conv1 = GATConv(embedding_size+1, hidden_channels)
        # self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.out1 = torch.nn.Linear(hidden_channels, hidden_channels//4)
        self.out2 = torch.nn.Linear(hidden_channels//4, num_classes)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        x = self.embedding(x)
        x = torch.cat((x, data.abundancies.unsqueeze(-1)), dim=-1)
        x = self.conv1(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_weights)
        x = global_mean_pool(x, data.batch)
        x = F.leaky_relu(self.out1(x))
        x = self.out2(x)
        return x
    

def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0
    predictions = []
    true_classes = []
    scores = []
    model.eval()
    for i, (batch, keys) in enumerate(tqdm(dataloader)):
        y = batch.y
        output = model(batch)
        scores.extend(F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        _, predicted = torch.max(output.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_classes.extend(y.cpu().numpy())
        total += y.size(0)
        correct += (predicted == y).sum().item()
    return predictions, true_classes, scores


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {'Accuracy': accuracy, 'F1 Score': f1}

def plot_and_save(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    plot_and_save(fig, 'evaluation_files/roc_curve.png')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plot_and_save(fig, 'evaluation_files/confusion_matrix.png')

def generate_pdf_report(metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Model Evaluation Report', ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Evaluation Metrics:', ln=True)
    pdf.set_font('Arial', '', 12)
    for key, value in metrics.items():
        pdf.cell(0, 10, f'{key}: {value:.4f}', ln=True)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 10 Important Nodes:', ln=True)
    pdf.set_font('Arial', '', 12)
    try:
        with open('evaluation_files/GNNExplainer.txt', 'r') as file:
            lines = file.readlines()
            # 10 first
            for line in lines[:10]:
                pdf.cell(0, 10, line.strip(), ln=True)
    except Exception as e:
        pdf.cell(0, 10, f'Error readine file: {e}', ln=True)

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Visualizations:', ln=True)
    pdf.image('evaluation_files/roc_curve.png', x=10, w=190)
    pdf.image('evaluation_files/confusion_matrix.png', x=10, w=190)
    pdf.image('evaluation_files/GNNExplainer_frequency_vs_score.png', x=10, w=190)
    pdf.output('evaluation_files/evaluation_report.pdf')

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
        return data, key


def load_dataloader():
    # load test_dataset
    test_inds = np.load('test_inds.npy')
    sampling_op_to_tensor = pkl.load(open('sampling_op_to_tensor.pkl', 'rb'))
    sampling_ops_to_remove = ['S05029820_20150602']
    for samp in sampling_ops_to_remove:
        sampling_op_to_tensor.pop(samp)
    sites = list(set([key.split('_')[0] for key in sampling_op_to_tensor.keys()]))
    test_sites = set([sites[i] for i in test_inds])
    test_dataset = GraphDataset_SplitSite('her_lvl_1', sampling_op_to_tensor, y='Nitrates_Status1Y', sites=test_sites)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    return test_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load('gcn_nitrates_model.pt', map_location=device)
    model.eval()
    model.to(device)
    
    test_dataloader = load_dataloader()

    predictions, true_values, scores = evaluate_model(model, test_dataloader, device)
    metrics = compute_metrics(true_values, predictions)
    print(metrics)
    
    plot_roc_curve(true_values, scores)
    plot_confusion_matrix(true_values, predictions)
    generate_pdf_report(metrics)

if __name__ == "__main__":
    main()
