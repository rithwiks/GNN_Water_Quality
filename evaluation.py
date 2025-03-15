import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from fpdf import FPDF
from tqdm import tqdm


def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0
    predictions = []
    true_classes = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            x, y, keys = data

            x, y = x.to(device), y.to(device)

            output = model(x.float())
            _, predicted = torch.max(output.data, 1)

            predictions.extend(predicted.cpu().numpy())
            true_classes.extend(y.cpu().numpy())

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return predictions, true_classes

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        'Accuracy': accuracy,
        'F1 Score': f1
    }

def plot_and_save(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

def plot_loss_accuracy(training_loss, validation_loss, training_accuracy, validation_accuracy):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(training_loss, label='Training Loss')
    ax[0].plot(validation_loss, label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Loss Curves')
    
    ax[1].plot(training_accuracy, label='Training Accuracy')
    ax[1].plot(validation_accuracy, label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Accuracy Curves')
    
    plot_and_save(fig, 'loss_accuracy.png')

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    
    plot_and_save(fig, 'roc_curve.png')

def generate_pdf_report(metrics):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
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
    pdf.cell(0, 10, 'Loss and Accuracy Curves:', ln=True)
    pdf.image('loss_accuracy.png', x=10, w=190)
    
    pdf.ln(10)
    pdf.cell(0, 10, 'ROC Curve:', ln=True)
    pdf.image('roc_curve.png', x=10, w=190)
    
    pdf.output('evaluation_report.pdf')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('diatom_model_complete.pth', map_location=device)
    model.to(device)
    
    from baseline import test_dataloader
    
    predictions, true_values = evaluate_model(model, test_dataloader, device)
    metrics = compute_metrics(true_values, predictions)
    print(metrics)
    
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    
    plot_loss_accuracy(training_loss, validation_loss, training_accuracy, validation_accuracy)
    
    plot_roc_curve(true_values, scores)

    generate_pdf_report(metrics)

if __name__ == "__main__":
    main()