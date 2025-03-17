import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fpdf import FPDF
from tqdm import tqdm
from baseline import Net
import sys
sys.modules['__main__'].Net = Net
from baseline import Net
import sys
sys.modules['__main__'].Net = Net


def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0
    predictions = []
    true_classes = []
    scores = []
    features = []
    with torch.no_grad():
        for data in tqdm(dataloader):
        for data in tqdm(dataloader):
            x, y, keys = data
            x, y = x.to(device), y.to(device)
            output = model(x.float())
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_classes.extend(y.cpu().numpy())
            total += y.size(0)
            correct += (predicted == y).sum().item()
            scores.extend(F.softmax(output, dim=1)[:, 1].cpu().numpy())
            features.extend(output.cpu().numpy())
    return predictions, true_classes, scores, features

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {'Accuracy': accuracy, 'F1 Score': f1}
    return {'Accuracy': accuracy, 'F1 Score': f1}

def plot_and_save(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

def plot_loss_accuracy():
    training_loss = np.load('training_loss.npy')
    training_accuracy = np.load('training_accuracy.npy')
    epochs = range(1, len(training_loss) + 1)
def plot_loss_accuracy():
    training_loss = np.load('training_loss.npy')
    training_accuracy = np.load('training_accuracy.npy')
    epochs = range(1, len(training_loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(epochs, training_loss, label='Training Loss')
    ax[0].plot(epochs, training_loss, label='Training Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')
    ax[0].set_title('Loss Curve')
    ax[0].legend()
    ax[1].plot(epochs, training_accuracy, label='Training Accuracy')
    ax[1].plot(epochs, training_accuracy, label='Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend()
    plot_and_save(fig, 'loss_accuracy.png')

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
    plot_and_save(fig, 'roc_curve.png')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plot_and_save(fig, 'confusion_matrix.png')

def plot_pca_tsne(features, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, ax=ax[0])
    ax[0].set_title('PCA Projection')
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, ax=ax[1])
    ax[1].set_title('t-SNE Projection')
    plot_and_save(fig, 'pca_tsne.png')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plot_and_save(fig, 'confusion_matrix.png')

def plot_pca_tsne(features, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, ax=ax[0])
    ax[0].set_title('PCA Projection')
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, ax=ax[1])
    ax[1].set_title('t-SNE Projection')
    plot_and_save(fig, 'pca_tsne.png')

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
    pdf.cell(0, 10, 'Visualizations:', ln=True)
    pdf.cell(0, 10, 'Visualizations:', ln=True)
    pdf.image('loss_accuracy.png', x=10, w=190)
    pdf.image('roc_curve.png', x=10, w=190)
    pdf.image('confusion_matrix.png', x=10, w=190)
    pdf.image('pca_tsne.png', x=10, w=190)
    pdf.output('evaluation_report.pdf')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('diatom_model_complete.pth', map_location=device)
    model.to(device)
    from baseline import test_dataloader
    predictions, true_values, scores, features = evaluate_model(model, test_dataloader, device)
    metrics = compute_metrics(true_values, predictions)
    print(metrics)
    plot_loss_accuracy()
    plot_loss_accuracy()
    plot_roc_curve(true_values, scores)
    plot_confusion_matrix(true_values, predictions)
    plot_pca_tsne(features, true_values)
    generate_pdf_report(metrics)

if __name__ == "__main__":
    main()
