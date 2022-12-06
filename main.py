import argparse
from typing import Dict

import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.metrics import Metrics
from model.unet import Unet, DEFAULT_UNET_LAYERS
from model.dice_loss import DiceLoss, DiceBCELoss
from datasets.dataset import RetinaSegmentationDataset
from utils.resultPrinter import ResultPrinter


def train_model(model, dataloader, criterion, optimizer, device):
    metrics_tracker = Metrics(device)
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
        # compute metrics
        metrics_tracker.calculate(lbl_pred, lbl)
        # Running tally
        train_running_loss += loss.item() * img.shape[0]
        # Backward step
        loss.backward()
        optimizer.step()

    # Compute the loss for this epoch
    train_loss = train_running_loss / (ind + 1)
    # Compute the metrics for this epoch
    metrics = metrics_tracker.get_mean_metrics(ind + 1)
    metrics['loss'] = train_loss
    return metrics


def eval_model(model, dataloader, criterion, device):
    metrics_tracker = Metrics(device)
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
            # compute metrics
            metrics_tracker.calculate(lbl_pred, lbl)
            # Running tally        
            eval_running_loss += loss.item() * img.shape[0]

    # Compute the loss for this epoch
    eval_loss = eval_running_loss / (ind + 1)
    # Compute the metrics for this epoch
    metrics = metrics_tracker.get_mean_metrics(ind + 1)
    metrics['loss'] = eval_loss
    return metrics


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
    parser.add_argument('--save-freq', default=1, type=int,
                        metavar='N', help='How frequent to save')
    parser.add_argument('--loss-function', nargs=1,
                        choices=['BCEWithLogitsLoss', 'CrossEntropyLoss', 'DiceLoss', 'DiceBCELoss'])
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout percent used in Unet Encoder')
    parser.add_argument('--unet-layers', default=None, type=str, help='Layer sizes for the U-Net as a string l1-l2-l3-...-ln')
    parser.add_argument('--scheduler', nargs=1, default='Fixed',
                        choices=['Fixed', 'CosineAnnealing', 'ReduceOnPlateau'])
    parser.add_argument('--anneal_tmax', default=10, type=int,
                        help='Cosine Annealing: Maximum number of iterations for cosine annealing')
    parser.add_argument('--anneal_eta', default=0, type=float,
                        help='Cosine Annealing: Minimum learning rate. Default: 0')
    
    args = parser.parse_args()

    # Get the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Determine the layer sizes of the U-Net
    unet_layers = DEFAULT_UNET_LAYERS
    if args.unet_layers:
        unet_layers = [int(x) for x in args.unet_layers.split("-")]

    # Initialize the model on the GPU
    model = Unet(dropout=args.dropout, hidden_channels=unet_layers).to(device)
    if args.load_encoder_weights:
        model.encoder.load_state_dict(torch.load(args.load_encoder_weights))
    elif args.load_bt_checkpoint:
        model.encoder.load_state_dict(torch.load(args.load_bt_checkpoint)["encoder"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define scheduler (if necessary)
    scheduler = None
    if args.scheduler[0] == 'CosineAnnealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.anneal_tmax, args.anneal_eta)
    elif args.scheduler[0] == 'ReduceOnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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

    # Create a descriptive name for the checkpoints
    temp_dict = dict(args._get_kwargs())
    descrip_name = ""
    for key in temp_dict.keys():
        if (key != "rootdir" and
            "load" not in key and
            "checkpoint" not in key and
            "workers" not in key and
            "save_freq" not in key):
            descrip_name += "--" + key + "=" + str(temp_dict[key])
    descrip_name = descrip_name.replace(' ', '_').replace('[', '').replace(']', '').replace('\'', '')

    # runs dict should be passed to each instance of a results printer. It is only appended to so should be thread safe.
    runs: Dict[str, Dict[str, float]] = {}
    # create a new results printer for each param setting tested
    result_printer = ResultPrinter(descrip_name, runs)

    epoch_pbar = tqdm(total=args.epochs, desc="Epochs")
    for i in range(args.epochs):

        train_metrics = train_model(model, training_dataloader, criterion, optimizer, device)
        result_printer.print(f'Training metrics: {str(train_metrics)}')
        train_loss = train_metrics['loss']

        validation_metrics = eval_model(model, validation_dataloader, criterion, device)
        result_printer.print(f'Validation metrics: {str(validation_metrics)}')
        validation_loss = validation_metrics['loss']

        result_printer.rankAndSave(validation_metrics)

        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        epoch_pbar.write("=" * 80)
        epoch_pbar.write("Epoch: {}".format(i))
        epoch_pbar.write("Train Loss : {:.4f}".format(train_loss))
        epoch_pbar.write("Validation Loss : {:.4f}".format(validation_loss))
        epoch_pbar.write("=" * 80)
        epoch_pbar.update(1)

        # Save plot of Train/Validation Loss Per Epoch
        result_printer.makePlots(training_losses, validation_losses, i)

        # Take appropriate scheduler step (if necessary)
        if args.scheduler[0] == 'CosineAnnealing':
            scheduler.step()
        elif args.scheduler[0] == 'ReduceOnPlateau':
            scheduler.step(validation_loss)

        if i % args.save_freq == 0:
            # save the model
            state = dict(epoch=i + 1,
                         model=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         unet_layer_sizes=unet_layers,
                         args=temp_dict)
            torch.save(state, args.checkpoint_dir / f'unet{descrip_name}.pth')

    result_printer.close()
