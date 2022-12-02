import argparse
import torch
import os
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
from tqdm import tqdm

from img_transform.transforms import EyeMaskCustomTransform, EyeDatasetCustomTransform
from model.unet import Unet
from model.dice_loss import DiceLoss, DiceBCELoss


IMG_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    EyeDatasetCustomTransform(mask_threshold=0.25),
])


LBL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    EyeMaskCustomTransform(mask_threshold=0.25),
])


class RetinaSegmentationDataset(Dataset):
    def __init__(self, rootdir: str,
                 basenames: List,
                 img_transforms: torch.nn.Module = IMG_TRANSFORMS,
                 lbl_transforms: torch.nn.Module = LBL_TRANSFORMS):
        self._rootdir = rootdir
        self._basenames = basenames
        self._img_transforms = img_transforms
        self._lbl_transforms = lbl_transforms
        
    def __len__(self) -> int:
        return len(self._basenames)
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        img_path = os.path.join(self._rootdir, "images", self._basenames[index])
        lbl_path = os.path.join(self._rootdir, "labels", self._basenames[index])
        
        with open(img_path, "rb") as f:
            img = pickle.load(f)
            # Apply the transforms for the image
            img = self._img_transforms(img)
            
        #'''
        with open(lbl_path, "rb") as f:
            lbl = pickle.load(f)
            # Apply the transforms for the labels
            lbl = self._lbl_transforms(lbl)
        #'''
            
        return img, lbl


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    train_running_loss = 0.0
    for ind, (img, lbl) in enumerate(tqdm(dataloader, desc="Training")):
        # Copy to device
        img = img.to(device)
        lbl = lbl.to(device)
        # Make the prediction
        lbl_pred = model(img)
        optimizer.zero_grad()
        # Compute loss
        loss = criterion(lbl_pred, lbl)
        # Running tally
        train_running_loss += loss.item() * img.shape[0]
        # Backward step
        loss.backward()
        optimizer.step()

    # Compute the loss for this epoch
    train_loss = train_running_loss / (ind + 1)
    return train_loss


def eval_model(model, dataloader, criterion, device):
    model.eval()
    eval_running_loss = 0.0
    with torch.no_grad():
        for ind, (img, lbl) in enumerate(tqdm(dataloader, desc="Validation")):
            # Copy to device
            img = img.to(device)
            lbl = lbl.to(device)
            # Make the prediction
            lbl_pred = model(img)
            # Compute loss        
            loss = criterion(lbl_pred, lbl)
            # Running tally        
            eval_running_loss += loss.item() * img.shape[0]

    # Compute the loss for this epoch
    eval_loss = eval_running_loss / (ind + 1)
    return eval_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument('--rootdir', type=Path, metavar='DIR',
                        help='path to dataset with Testing, Training, and Validation directories')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--load-encoder-weights', default=None, type=Path,
                        metavar='DIR', help='Weights to load for the encoder')
    parser.add_argument('--load-bt-checkpoint', default=None, type=Path,
                        metavar='DIR', help='Checkpoint weights to load for the encoder')
    parser.add_argument('--save-freq', default=5, type=int,
                        metavar='N', help='How frequent to save')
    parser.add_argument('--loss-function', nargs=1,
                        choices=['BCEWithLogitsLoss', 'CrossEntropyLoss', 'DiceLoss', 'DiceBCELoss'])
    
    args = parser.parse_args()

    # Get the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Initialize the model on the GPU
    model = Unet().to(device)
    if args.load_encoder_weights:
        model.encoder.load_state_dict(torch.load(args.load_encoder_weights))
    elif args.load_bt_checkpoint:
        model.encoder.load_state_dict(torch.load(args.load_bt_checkpoint)["encoder"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Select the Loss function
    loss_functions = {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "DiceLoss": DiceLoss(),
        "DiceBCELoss": DiceBCELoss()
    }
    criterion = loss_functions[args.loss_function[0]]

    # Load the training datasets
    training_path = os.path.join(args.rootdir, "Training")
    training_file_basenames = os.listdir(os.path.join(training_path, "images"))
    training_dataset = RetinaSegmentationDataset(training_path, training_file_basenames)
    training_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle=True)

    # Load the validation datasets
    validation_path = os.path.join(args.rootdir, "Validation")
    validation_file_basenames = os.listdir(os.path.join(validation_path, "images"))
    validation_dataset = RetinaSegmentationDataset(validation_path, validation_file_basenames)
    validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle=False)
    
    # Train / Val loop
    training_losses = []
    validation_losses = []

    epoch_pbar = tqdm(total=args.epochs, desc="Epochs")
    for i in range(args.epochs):
        train_loss = train_model(model, training_dataloader, criterion, optimizer, device)
        validation_loss = eval_model(model, validation_dataloader, criterion, device)
        training_losses.append(train_loss)
        epoch_pbar.write("=" * 80)
        epoch_pbar.write("Epoch: {}".format(i))
        epoch_pbar.write("Train Loss : {:.4f}".format(train_loss))
        epoch_pbar.write("Validation Loss : {:.4f}".format(validation_loss))
        epoch_pbar.write("=" * 80)
        epoch_pbar.update(1)

        if i % args.save_freq == 0:
            # save the model
            state = dict(epoch=i + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'image-segmentation-checkpoint.pth')
