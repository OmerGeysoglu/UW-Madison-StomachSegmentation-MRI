import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''
    
    # Apply sigmoid activation and thresholding to the model outputs
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()

    # Convert tensors to NumPy arrays
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = outputs.cpu().numpy()

    # Create a grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.ravel()

    # Display the first 9 samples from the batch
    for i in range(3):
        # Display the input image
        axs[i].imshow(images[i, 0], cmap='bone')
        axs[i].set_title('Input Image')
        axs[i].axis('off')

        # Display the ground truth mask
        axs[i + 3].imshow(masks[i, 0], cmap='bone')
        axs[i + 3].set_title('Ground Truth Mask')
        axs[i + 3].axis('off')

        # Display the predicted mask
        axs[i + 6].imshow(outputs[i, 0], cmap='bone')
        axs[i + 6].set_title('Predicted Mask')
        axs[i + 6].axis('off')
    
    # Save the visualization as a PNG file
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "pred_images" ,f'epoch_{epoch}_batch_{batch_idx}_predictions.png'))
    plt.close()

def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''
    # Create a range of epoch numbers
    epochs = range(1, len(train_loss_history) + 1)

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()

    # Save the plot as a JPEG file
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plots' ,f'{args.exp_id}_loss_curve.jpg'))
    plt.close()


def plot_metric(x, label, plot_dir, args, metric):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    # Create a range of epoch numbers
    epochs = range(1, len(x) + 1)

    # Plot the metric curve
    plt.figure(figsize=(10, 6))
    x = x.cpu().numpy()
    plt.plot(epochs, x, label=label, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(f'{label} Curve')
    plt.legend()

    # Save the plot as a JPEG file
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plots' , f'{args.exp_id}_{metric}_curve.jpg'))
    plt.close()