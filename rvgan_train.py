import argparse
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
from torchvision.transforms import Resize

from model.metrics import Metrics
from model.rvgan import RVGAN
from losses.losses import RVGANLoss
from datasets.dataset import RetinaSegmentationDataset


def train_model(
    model: RVGAN,
    dataloader: RetinaSegmentationDataset,
    optimizers: Dict[str, torch.optim.Optimizer],
    device: str
):
    criterion = RVGANLoss()
    metrics_tracker = Metrics(device)
    train_running_loss = 0.0    
    for ind, (img, lbl) in enumerate(tqdm(dataloader, desc="Training")):
        # Copy to device
        img = img.to(device)
        lbl = lbl.to(device)
        # Decimate image and label
        decimated_shape = [int(d / model.decimination_factor) for d in img.shape[2:]]
        resizer = Resize(size=decimated_shape)
        decimated_img = resizer(img)
        decimated_lbl = resizer(lbl)

        # Train only the discriminator
        model.eval()
        model.coarse_discriminator.train()
        model.fine_discriminator.train()
        model.generators.coarse_generator.eval()
        model.generators.fine_generator.eval()
        # Run this batch through twice to train discriminator
        for _ in range(2):
            # Create the fake fine images
            coarse_generator_out, fine_generator_out = model.generators(img, decimated_x=decimated_img)
            # fine real discriminator out
            optimizers["Fine_Discriminator"].zero_grad()
            fine_real_disc_out, _ = model.fine_discriminator(img, vessel_labels=lbl)
            d_loss1 = torch.nn.MSELoss()(fine_real_disc_out, -torch.ones_like(fine_real_disc_out))
            d_loss1.backward()
            optimizers["Fine_Discriminator"].step()

            optimizers["Fine_Discriminator"].zero_grad()
            fine_fake_disc_out, _ = model.fine_discriminator(img, vessel_labels=fine_generator_out)
            d_loss2 = torch.nn.MSELoss()(fine_fake_disc_out, torch.ones_like(fine_fake_disc_out))
            d_loss2.backward()
            optimizers["Fine_Discriminator"].step()

            # Create the fake coarse images
            coarse_generator_out = model.generators.coarse_generator(decimated_img)

            # coarse real discriminator out
            optimizers["Coarse_Discriminator"].zero_grad()
            coarse_real_disc_out, _ = model.coarse_discriminator(decimated_img, vessel_labels=decimated_lbl)
            d_loss3 = torch.nn.MSELoss()(coarse_real_disc_out, -torch.ones_like(coarse_real_disc_out))
            d_loss3.backward()
            optimizers["Coarse_Discriminator"].step()

            optimizers["Coarse_Discriminator"].zero_grad()
            coarse_fake_disc_out, _ = model.coarse_discriminator(decimated_img, vessel_labels=coarse_generator_out)
            d_loss4 = torch.nn.MSELoss()(coarse_fake_disc_out, torch.ones_like(coarse_fake_disc_out))
            d_loss4.backward()
            optimizers["Coarse_Discriminator"].step()

        # Train Coarse Geneator
        model.eval()
        model.coarse_discriminator.eval()
        model.fine_discriminator.eval()
        model.generators.coarse_generator.train()
        model.generators.fine_generator.eval()

        optimizers["Coarse_Generator"].zero_grad()
        pred_decimated_vessels = model.generators.coarse_generator(decimated_img)
        g_global_loss = torch.nn.MSELoss()(pred_decimated_vessels, decimated_lbl)
        g_global_loss.backward()
        optimizers["Coarse_Generator"].step()

        # Train Fine Generator
        model.eval()
        model.coarse_discriminator.eval()
        model.fine_discriminator.eval()
        model.generators.coarse_generator.eval()
        model.generators.fine_generator.train()

        optimizers["Fine_Generator"].zero_grad()
        _, pred_vessels = model.generators(img, decimated_x=decimated_img)
        g_local_loss = torch.nn.MSELoss()(pred_vessels, lbl)
        g_local_loss.backward()
        optimizers["Fine_Generator"].step()

        # Train RVGAN
        model.train()
        model.coarse_discriminator.train()
        model.fine_discriminator.train()
        model.generators.coarse_generator.eval()
        model.generators.fine_generator.eval()
        # Make the prediction
        output_dict = model(img, vessel_labels=lbl)
        optimizers["RVGAN"].zero_grad()
        # Compute loss
        train_running_loss = criterion.compute_rvgan_loss(output_dict)
        # compute metrics
        metrics_tracker.calculate(output_dict["Fake"]["Fine Discriminator Out"], lbl)
        # Running tally
        train_running_loss += train_running_loss * img.shape[0]
        # Backward step
        train_running_loss.backward()
        optimizers["RVGAN"].step()

        break

    # Compute the loss for this epoch
    train_loss = train_running_loss / (ind + 1)
    # Compute the metrics for this epoch
    print(f'Training metrics: {str(metrics_tracker.get_mean_metrics(ind + 1))}')
    return train_loss.detach().cpu().numpy()


def eval_model(model: RVGAN, dataloader, device):
    criterion = torch.nn.MSELoss()
    metrics_tracker = Metrics(device)
    model.eval()
    eval_running_loss = 0.0
    with torch.no_grad():
        for ind, (img, lbl) in enumerate(tqdm(dataloader, desc="Validation")):
            # Copy to device
            img = img.to(device)
            lbl = lbl.to(device)
            # Make the prediction
            _, lbl_pred = model.generators(img)
            # Compute loss        
            loss = criterion(lbl_pred, lbl)
            # compute metrics
            metrics_tracker.calculate(lbl_pred, lbl)
            # Running tally        
            eval_running_loss += loss.item() * img.shape[0]

            break

    # Compute the loss for this epoch
    eval_loss = eval_running_loss / (ind + 1)
    # Compute the metrics for this epoch
    print(f'Validation metrics: {str(metrics_tracker.get_mean_metrics(ind + 1))}')
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
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--load-encoder-weights', default=None, type=Path,
                        metavar='DIR', help='Weights to load for the encoder')
    parser.add_argument('--load-bt-checkpoint', default=None, type=Path,
                        metavar='DIR', help='Checkpoint weights to load for the encoder')
    parser.add_argument('--save-freq', default=1, type=int,
                        metavar='N', help='How frequent to save')
    #parser.add_argument('--dropout', default=0.2, type=float, help='dropout percent used in Unet Encoder')
    #parser.add_argument('--anneal_tmax', default=10, type=int,
    #                    help='Cosine Annealing: Maximum number of iterations for cosine annealing')
    #parser.add_argument('--anneal_eta', default=0, type=float,
    #                    help='Cosine Annealing: Minimum learning rate. Default: 0')
    
    args = parser.parse_args()

    # Get the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    """
    # Determine the layer sizes of the U-Net
    unet_layers = DEFAULT_UNET_LAYERS
    if args.unet_layers:
        unet_layers = [int(x) for x in args.unet_layers.split("-")]
    """

    # Initialize the model on the GPU
    model = RVGAN(num_channels_in=4).to(device)
    """
    if args.load_encoder_weights:
        model.encoder.load_state_dict(torch.load(args.load_encoder_weights))
    elif args.load_bt_checkpoint:
        model.encoder.load_state_dict(torch.load(args.load_bt_checkpoint)["encoder"])
    """
    optimizers = {
        "Fine_Discriminator": torch.optim.Adam(
            model.fine_discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999]
        ),
        "Coarse_Discriminator": torch.optim.Adam(
            model.coarse_discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999]
        ),
        "Fine_Generator": torch.optim.Adam(
            model.generators.fine_generator.parameters(), lr=0.0002, betas=[0.5, 0.999]
        ),
        "Coarse_Generator": torch.optim.Adam(
            model.generators.coarse_generator.parameters(), lr=0.0002, betas=[0.5, 0.999]
        ),
        "RVGAN": torch.optim.Adam(
            model.parameters(), lr=0.0002, betas=[0.5, 0.999]
        )
    }

    """
    # Define scheduler (if necessary)
    scheduler = None
    if args.scheduler[0] == 'CosineAnnealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.anneal_tmax, args.anneal_eta)
    elif args.scheduler[0] == 'ReduceOnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    """

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

    epoch_pbar = tqdm(total=args.epochs, desc="Epochs")
    for i in range(args.epochs):
        train_loss = train_model(model, training_dataloader, optimizers, device)
        validation_loss = eval_model(model, validation_dataloader, device)
        training_losses.append(train_loss)
        #validation_losses.append(validation_loss)
        epoch_pbar.write("=" * 80)
        epoch_pbar.write("Epoch: {}".format(i))
        epoch_pbar.write("Train Loss : {:.4f}".format(train_loss))
        #epoch_pbar.write("Validation Loss : {:.4f}".format(validation_loss))
        epoch_pbar.write("=" * 80)
        epoch_pbar.update(1)

        # Save plot of Train/Validation Loss Per Epoch
        plt.clf()
        plt.plot(range(len(training_losses)),training_losses,'r')
        #plt.plot(range(len(validation_losses)),validation_losses,'b') 
        plt.legend(['Training Loss','Validation Loss'])
        plt.title('Training and Validation Loss for Unet')
        plt.savefig('plot' + str(i) + '.png')

        """
        # Take appropriate scheduler step (if necessary)
        if args.scheduler[0] == 'CosineAnnealing':
            scheduler.step()
        elif args.scheduler[0] == 'ReduceOnPlateau':
            scheduler.step(validation_loss)
        """

        if i % args.save_freq == 0:
            # save the model
            state = dict(epoch=i + 1,
                         model=model.state_dict(),
                         args=temp_dict)
            torch.save(state, args.checkpoint_dir / f'rvgan{descrip_name}.pth')
