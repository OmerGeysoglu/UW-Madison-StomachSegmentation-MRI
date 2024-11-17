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
from losses.dice_loss import DiceLoss

def test_model(model, test_dataloader, args, save_path):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - test_dataloader (DataLoader): DataLoader for the test dataset.
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

    # Initialize an empty list to store the Dice coefficients
    dice_loss_fn = DiceLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    dice_coef_history = []
    total_dice_coef = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(test_dataloader):
            images, masks = images.to(args.device), masks.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice_coef = 1 - dice_loss_fn(outputs, masks)
            dice_coef_history.append(dice_coef.item())
            total_loss += loss.item()
            total_dice_coef += dice_coef.item()
        
        avg_dice_coef = total_dice_coef / len(test_dataloader)
        avg_loss = total_loss / len(test_dataloader)
        print(f'Average Dice coefficient: {avg_dice_coef:.4f}')
        print(f'Average Loss: {avg_loss:.4f}')

        # Save the Dice coefficient history plot
        plot_metric(dice_coef_history, 'Test Dice Coefficient', save_path, args, 'test_dice_coef')





if __name__ == '__main__':

    args = test_arg_parser()
    save_path = os.path.join("results", args.exp_id)
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="madison-stomach-data", 
                            mode="test")

    test_dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    # Define and load your model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    test_model(model=model,
               test_dataloader=test_dataloader,
                args=args,
                save_path=save_path)