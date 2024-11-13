import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import test_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import DiceLoss

def test_model(model, args, save_path):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - args (argparse.Namespace): Parsed arguments for device, batch size, etc.
    - save_path (str): Directory where results (e.g., metrics plot) will be saved.
    
    Functionality:
    - Sets the model to evaluation mode and iterates over the test dataset.
    - Computes the Dice score for each batch and calculates the average.
    - Saves a plot of the Dice coefficient history.
    '''
    
    # Set the model to evaluation mode
    model.eval()
    model.to(args.device)

    # Initialize lists to store Dice scores
    dice_scores = []

    # Iterate over the test dataset
    for images, masks in tqdm(test_dataloader):
        # Load the batch to the device
        images = images.to(args.device)
        masks = masks.to(args.device)

        # Perform inference
        with torch.no_grad():
            pred_mask = model(images)

        # Compute the Dice score
        dice_loss = DiceLoss()
        dice_score = dice_loss(pred_mask, images)
        dice_score = 1 - dice_score
        dice_scores.append(dice_score.detach().cpu().item())

    # Calculate the average Dice score
    avg_dice_score = sum(dice_scores) / len(dice_scores)
    print(f'Average Dice Score: {avg_dice_score}')

    # Save the Dice coefficient history plot
    plot_metric(dice_scores, label="Dice Coefficient", plot_dir=save_path, args=args, metric='dice_coeff')



if __name__ == '__main__':

    args = test_arg_parser()
    save_path = "results/"
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="madison-stomach-data", 
                            mode="test")

    test_dataloader = DataLoader(dataset, batch_size=10)

    # Define and load your model
    model = UNet(in_channels=1, out_channels=1)
    model = torch.load(args.model_path, map_location=args.device)

    test_model(model=model,
                args=args,
                save_path=save_path)