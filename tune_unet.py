import argparse
import torch
import os
from torch.utils.data import Dataset
from typing import Dict
from pathlib import Path
from tqdm import tqdm
from model.metrics import Metrics
from model.unet import Unet, DEFAULT_UNET_LAYERS
from model.dice_loss import DiceLoss, DiceBCELoss
from datasets.dataset import RetinaSegmentationDataset
from utils.resultPrinter import ResultPrinter
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from multiprocessing import cpu_count


# NOTE: Referenced https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html when creating this script


# runs dict should be passed to each instance of a results printer. It is only appended to so should be thread safe.
runs: Dict[str, Dict[str, float]] = {}


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


def train_unet(config, workers, epochs, data_directory, load_encoder_weights, load_bt_checkpoint):
    # Get the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Determine the layer sizes of the U-Net
    if config["unet_layers"] != DEFAULT_UNET_LAYERS:
        unet_layers = [int(x) for x in config["unet_layers"].split("-")]
    else:
        unet_layers = DEFAULT_UNET_LAYERS

    # Initialize the model on the GPU
    model = Unet(dropout=config["dropout"], hidden_channels=unet_layers).to(device)
    if load_encoder_weights:
        model.encoder.load_state_dict(torch.load(load_encoder_weights))
    elif load_bt_checkpoint:
        model.encoder.load_state_dict(torch.load(load_bt_checkpoint)["encoder"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Define scheduler (if necessary)
    select_scheduler = None
    if config["scheduler"] == 'CosineAnnealing':
        select_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # Hard coding T_max
    elif config["scheduler"] == 'ReduceOnPlateau':
        select_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Select the Loss function
    loss_functions = {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "DiceLoss": DiceLoss(),
        "DiceBCELoss": DiceBCELoss()
    }
    criterion = loss_functions[config["loss_func"]]

    # Load the training datasets
    training_path = os.path.join(data_directory, "Training")
    training_file_basenames = os.listdir(os.path.join(training_path, "images"))
    training_dataset = RetinaSegmentationDataset(training_path, training_file_basenames)
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=config["batch_size"], num_workers=workers,
        pin_memory=True, shuffle=True)

    # Load the validation datasets
    validation_path = os.path.join(data_directory, "Validation")
    validation_file_basenames = os.listdir(os.path.join(validation_path, "images"))
    validation_dataset = RetinaSegmentationDataset(validation_path, validation_file_basenames)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=config["batch_size"], num_workers=workers,
        pin_memory=True, shuffle=False)

    # Train / Val loop
    training_losses = []
    validation_losses = []

    # Create a descriptive name for the checkpoints
    temp_dict = dict(config)
    descrip_name = ""
    for key in temp_dict.keys():
        descrip_name += "--" + key + "=" + str(temp_dict[key])
    descrip_name = descrip_name.replace(' ', '_').replace('[', '').replace(']', '').replace('\'', '')

    # create a new results printer for each param setting tested
    result_printer = ResultPrinter(descrip_name, runs)

    epoch_pbar = tqdm(total=epochs, desc="Epochs")
    for i in range(epochs):
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
        if config["scheduler"] == 'CosineAnnealing':
            select_scheduler.step()
        elif config["scheduler"] == 'ReduceOnPlateau':
            select_scheduler.step(validation_loss)

        # Communicate validation loss to ray-tune
        tune.report(loss=validation_loss)

    result_printer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument('--rootdir', type=Path, metavar='DIR',
                        help='path to dataset with Testing, Training, and Validation directories')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--load-encoder-weights', default=None, type=Path,
                        metavar='DIR', help='Weights to load for the encoder')
    parser.add_argument('--load-bt-checkpoint', default=None, type=Path,
                        metavar='DIR', help='Checkpoint weights to load for the encoder')
    parser.add_argument('--max_num_epochs', default=10, type=int, metavar='N',
                        help='https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html')
    parser.add_argument('--num_samples', default=10, type=int, metavar='N',
                        help='https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html')
    parser.add_argument('--gpus_per_trial', default=0, type=float, metavar='N',
                        help='https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html')

    args = parser.parse_args()

    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "loss_func": tune.choice(['BCEWithLogitsLoss', 'CrossEntropyLoss', 'DiceLoss', 'DiceBCELoss']),
        "dropout": tune.choice([0.2, 0.5]),
        "scheduler": tune.choice(['Fixed', 'CosineAnnealing', 'ReduceOnPlateau']),
        "unet_layers": tune.choice(["16-32", "16-32-64", "16-32-64-128"])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["lr", "batch_size", "loss_func", "dropout", "scheduler"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train_unet, workers=args.workers, epochs=args.epochs, data_directory=args.rootdir,
                load_encoder_weights=args.load_encoder_weights, load_bt_checkpoint=args.load_bt_checkpoint),
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_concurrent_trials=1,
        resources_per_trial={"cpu": cpu_count(), "gpu": 1}
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
