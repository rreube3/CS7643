import argparse
import random
from threading import Thread
from time import sleep

import torch
import os
import numpy as np
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
from multiprocessing import cpu_count, Manager, Process, freeze_support


# NOTE: Referenced https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html when creating this script


# runs dict should be passed to each instance of a results printer. It is only appended to so should be thread safe.

def train_model(model, dataloader, criterion, optimizer, device, thread_num):
    metrics_tracker = Metrics(device)
    model.train()
    train_running_loss = 0.0
    for ind, (img, lbl) in enumerate(tqdm(dataloader, desc="Training - Thread {}".format(thread_num))):
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


def eval_model(model, dataloader, criterion, device, thread_num):
    metrics_tracker = Metrics(device)
    model.eval()
    eval_running_loss = 0.0
    with torch.no_grad():
        for ind, (img, lbl) in enumerate(tqdm(dataloader, desc="Validation - Thread {}".format(thread_num))):
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


class TrainingRunnable:

    def __init__(self, paramList, run_name = None):
        self.paramList = paramList
        self._run_name = run_name

    def run(self, thread_num, workers, epochs, data_directory, load_encoder_weights, load_bt_checkpoint):
        for config in tqdm(self.paramList, desc="Config Combo - Thread {}".format(thread_num)):

            # Create a descriptive name for the checkpoints
            temp_dict = dict(config)
            descrip_name = ""
            for key in temp_dict.keys():
                descrip_name += "--" + key + "=" + str(temp_dict[key])
            descrip_name = descrip_name.replace(' ', '_').replace('[', '').replace(']', '').replace('\'', '')

            try:

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
                    select_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)  # Hard coding T_max
                elif config["scheduler"] == 'ReduceOnPlateau':
                    select_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

                # Select the Loss function
                loss_functions = {
                    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
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

                # create a new results printer for each param setting tested
                result_printer = ResultPrinter(descrip_name, runs, run_name=self._run_name)

                epoch_pbar = tqdm(total=epochs, desc="Epochs - Thread {}".format(thread_num))

                prev_validation_loss = None

                for i in range(epochs):
                    train_metrics = train_model(model, training_dataloader, criterion, optimizer, device, thread_num)
                    result_printer.print(f'Training metrics: {str(train_metrics)}')
                    train_loss = train_metrics['loss']

                    validation_metrics = eval_model(model, validation_dataloader, criterion, device, thread_num)
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

                    if prev_validation_loss is not None:
                        if np.abs(prev_validation_loss - validation_loss) / prev_validation_loss < 0.01:
                            break

                    prev_validation_loss = validation_loss
                del model
                result_printer.close()
            except Exception as err:
                print(f'{descrip_name} failed due to {str(err)} sleeping 5 seconds')
                sleep(5)


if __name__ == '__main__':
    runs: Dict[str, Dict[str, float]] = Manager().dict()
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
    parser.add_argument('--num-cpus', default=1, type=int, metavar='N',
                        help='Num CPUs')
    parser.add_argument('--run-name', default=None, type=str,
                        help='Run Name')

    args = parser.parse_args()

    total_args = {
        "lr": [1e-4, 1e-3, 1e-2],
        "batch_size": [16, 32, 64],
        "loss_func": ['BCEWithLogitsLoss', 'DiceLoss', 'DiceBCELoss'],
        "dropout": [0.2, ],
        "scheduler": ['Fixed', 'CosineAnnealing', 'ReduceOnPlateau'],
        "unet_layers": ["16-32", "16-32-64", "16-32-64-128", "64-128", "64-128-256"]#, "64-128-256-512"]
    }

    numComb = 0
    for k in total_args.keys():
        numComb = numComb + len(total_args[k]) if numComb == 0 else numComb * len(total_args[k])

    denom = min(args.num_cpus, numComb)

    print(f'testing {numComb} hyper-parameter combinations on {denom} processors')
    allCombs = []
    for i in total_args['lr']:
        for j in total_args['batch_size']:
            for k in total_args['loss_func']:
                for a in total_args['dropout']:
                    for b in total_args['scheduler']:
                        for c in total_args['unet_layers']:
                            allCombs.append(
                                {
                                    "lr": i,
                                    "batch_size": j,
                                    "loss_func": k,
                                    "dropout": a,
                                    "scheduler": b,
                                    "unet_layers": c
                                }
                            )

    sizeParamsToTry = []
    paramsToTry = []

    random.shuffle(allCombs)
    sz = numComb // denom
    for i in range(denom):
        if i == denom - 1:
            paramsToTry.append(allCombs[i * sz:])
        else:
            paramsToTry.append(allCombs[i * sz: (i + 1) * sz])
    print(paramsToTry)

    waitForThese = []

    for i in range(denom):
        t = TrainingRunnable(paramsToTry[i], run_name=args.run_name)
        p = Thread(target=t.run,
                   args=(i, args.workers, args.epochs, args.rootdir, args.load_encoder_weights, args.load_bt_checkpoint))
        p.start()
        waitForThese.append(p)

    for i in waitForThese:
        while True:
            i.join(timeout=10)
            if not i.is_alive():
                break
    print('done')
