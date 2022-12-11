import argparse
import torch
import os
#from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
#from tqdm import tqdm

#from model.metrics import Metrics
from model.unet import Unet, DEFAULT_UNET_LAYERS
from model.dice_loss import DiceLoss, DiceBCELoss
from datasets.dataset import RetinaSegmentationDataset
#from utils.resultPrinter import ResultPrinter

#-------added the following-----------
from matplotlib.ticker import MaxNLocator
import numpy as np
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
#-----------

import warnings
warnings.filterwarnings('ignore')

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
    parser.add_argument('--run-name', default=None, type=str,
                        help='Run Name')
    parser.add_argument('--k_folds', default=2, type=int,
                        help='k-fold cross-validation k>1. Default: 2')
    args = parser.parse_args()


    # Get the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Determine the layer sizes of the U-Net
    unet_layers = DEFAULT_UNET_LAYERS
    if args.unet_layers:
        unet_layers = [int(x) for x in args.unet_layers.split("-")]


    # Define scheduler (if necessary)
    scheduler = None

    # Select the Loss function
    loss_functions = {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "DiceLoss": DiceLoss(),
        "DiceBCELoss": DiceBCELoss()
    }
    criterion = loss_functions[args.loss_function[0]]

    #k-fold Cross Validation kicks in here
    training_path = os.path.join(args.rootdir, "Training")
    training_file_basenames = os.listdir(os.path.join(training_path, "images"))
    training_dataset = RetinaSegmentationDataset(training_path, training_file_basenames)
    validation_path = os.path.join(args.rootdir, "Validation")
    validation_file_basenames = os.listdir(os.path.join(validation_path, "images"))
    validation_dataset = RetinaSegmentationDataset(validation_path,validation_file_basenames)
    dataset = ConcatDataset([training_dataset, validation_dataset])
    kfold = KFold(n_splits=args.k_folds, shuffle=True)

     # Create a descriptive name for the checkpoints
#    temp_dict = dict(args._get_kwargs())
#    descrip_name = ""
#    for key in temp_dict.keys():
#        if (key != "rootdir" and
#            "load" not in key and
#            "checkpoint" not in key and
#            "workers" not in key and
#            "save_freq" not in key):
#            descrip_name += "--" + key + "=" + str(temp_dict[key])
#    descrip_name = descrip_name.replace(' ', '_').replace('[', '').replace(']', '').replace('\'', '')

    # runs dict should be passed to each instance of a results printer. 
    #It is only appended to so should be thread safe.
#    runs: Dict[str, Dict[str, float]] = {}
    # create a new results printer for each param setting tested
    #result_printer = ResultPrinter(descrip_name, runs, run_name=args.run_name)
    #result_printer = ResultPrinter(descrip_name, runs, time_label=args.run_name)

    #Initialize arrays to hold training and validation losses to averate at the end
    training_losses = np.zeros((args.epochs,args.k_folds))
    validation_losses = np.zeros((args.epochs,args.k_folds))
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset)):

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

        # Define data loaders for training and validation data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                      num_workers=args.workers, pin_memory=True,
                      sampler=train_subsampler)
        validationloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                      num_workers=args.workers, pin_memory=True,
                      sampler=validation_subsampler)

        model = Unet(dropout=args.dropout, hidden_channels=unet_layers).to(device)
#        if args.load_encoder_weights:
#            model.encoder.load_state_dict(torch.load(args.load_encoder_weights))
#        elif args.load_bt_checkpoint:
#            model.encoder.load_state_dict(torch.load(args.load_bt_checkpoint)["encoder"])

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.scheduler[0] == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
                            args.anneal_tmax, args.anneal_eta)
        elif args.scheduler[0] == 'ReduceOnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


        for i in range(0,args.epochs):
            #Training
            #metrics_tracker = Metrics(device)
            model.train()
            train_running_loss = 0.0
            for ind, (img, lbl) in enumerate(trainloader):
                img = img.to(device)
                lbl = lbl.to(device)
                lbl_pred = model(img)
                optimizer.zero_grad()
                loss = criterion(lbl_pred, lbl)
                #metrics_tracker.calculate(lbl_pred, lbl)
                train_running_loss += loss.item() * img.shape[0]
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()*img.shape[0]
            train_loss = train_running_loss / (ind + 1)
            #train_metrics = metrics_tracker.get_mean_metrics(ind + 1)
            #train_metrics['loss'] = train_loss
#            result_printer.print(f'Training metrics: {str(train_metrics)}')

            #Validation
            #metrics_tracker = Metrics(device)
            model.eval()
            eval_running_loss = 0.0
            with torch.no_grad():
                for ind, (img, lbl) in enumerate(validationloader):
                    img = img.to(device)
                    lbl = lbl.to(device)
                    lbl_pred = model(img)
                    loss = criterion(lbl_pred, lbl)
                    #metrics_tracker.calculate(lbl_pred, lbl)
                    eval_running_loss += loss.item() * img.shape[0]
            valid_loss = eval_running_loss / (ind + 1)
            #validation_metrics = metrics_tracker.get_mean_metrics(ind + 1)
            #validation_metrics['loss'] = valid_loss
            #result_printer.print(f'Validation metrics: {str(validation_metrics)}')
            #result_printer.rankAndSave(validation_metrics)

            print('Fold = ',fold+1, ', Epoch = ',i+1, \
                   'Training Loss = ',train_loss, \
                   'Validation_loss= ',valid_loss)
            training_losses[i][fold] = train_loss
            validation_losses[i][fold] = valid_loss

#            result_printer.makePlots(training_losses, validation_losses, i)

            # Take appropriate scheduler step (if necessary)
            if args.scheduler[0] == 'CosineAnnealing':
                scheduler.step()
            elif args.scheduler[0] == 'ReduceOnPlateau':
                scheduler.step(valid_loss)

#            if i % args.save_freq == 0:
#                # save the model
#                state = dict(fold = fold+1,
#                         epoch=i + 1,
#                         model=model.state_dict(),
#                         optimizer=optimizer.state_dict(),
#                         unet_layer_sizes=unet_layers,
#                         args=temp_dict)
#                torch.save(state, args.checkpoint_dir / f'unet{descrip_name}.pth')
        break

    #T = np.array(training_losses).mean(axis=1)
    #V = np.array(validation_losses).mean(axis=1)
    T = np.array(training_losses).max(axis=1)
    V = np.array(validation_losses).max(axis=1)

    #for i in range(args.epochs):
    for i in range(args.epochs-1,args.epochs):
        plt.clf()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1,i+2),T[0:i+1],'r')
        plt.plot(range(1,i+2),V[0:i+1],'b') 
        plt.legend(['Training Loss','Validation Loss'])
        plt.title('K-Fold Cross-Validation for Unet, K='+str(args.k_folds))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        cwd = os.getcwd().replace('\\','/')
        plt.savefig(cwd + '/plots/plot_' + str(args.k_folds) + 'folds_' + str(i+1) + 'epochs' + '.png')
        #result_printer.makePlots(T[0:i+1], V[0:i+1], i)

#    result_printer.close()


