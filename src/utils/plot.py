import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(all_labels, all_preds, categories,run_name, dir_path):
    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    
    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    plt.savefig(os.path.join(f'{dir_path}/{run_name}/results', 'cm.png'))
    plt.close()
    
def save_plot_loss_acc(train_accuracies, val_accuracies, run_name,dir_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(os.path.join(f'{dir_path}/{run_name}/results', 'accuracy_plot.png'))
    plt.close()
