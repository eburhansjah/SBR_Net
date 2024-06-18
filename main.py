import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import yaml
import time
import wandb
import pprint

from loss import PinballLoss
from src.train_and_validate import train_and_validate
from torch.utils.data import DataLoader, random_split
from data_loader import read_pq_file, TiffDataset
from src.SBR_NET import SBR_Net, kaiming_he_init


def main():
    start_time = time.time()

    # Initializing wanb sweep
    run = wandb.init(project="SBR_Net_eburhan", entity="cisl-bu")
    config = run.config
    
    # Flag on training one or multiple samples (Default: mult. samples)
    train_single_sample = config.get("train_single_sample", False)

    # Flag on whether or not to use mixed precision (Default: false)
    use_mixed_precision = config.get("use_mixed_precision")

    # Model parameters
    batch_size = config.get("batch_size")
    in_channels_rfv = config.get("in_channels_rfv")
    in_channels_stack = config.get("in_channels_stack")
    num_blocks = config.get("num_blocks")

    # Optimizer parameters
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")

    # Scheduler parameters
    T_max = config.get("T_max")
    eta_min = config.get("eta_min")
    last_epoch = config.get("last_epoch")

    num_epochs = config.get("num_epochs")

    ##Vascular dataset
    # input_file_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/training_data/UQ/vasc/15/metadata.pq"
    
    # Non-vascular dataset
    input_file_path = "metadata.pq"

    # Reading .tiff files
    stack_paths, rfv_paths, truth_paths = read_pq_file(input_file_path, one_sample=train_single_sample)

    # Converting .tiff files into tensor
    dataset = TiffDataset(stack_paths=stack_paths, rfv_paths=rfv_paths, truth_paths=truth_paths)

    if train_single_sample:
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        # Splitting dataset randomly into training and validation (80% and 20% respectively)
        train_sz = round(0.8 * len(dataset))
        val_sz = round(0.2 * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
        
        # Creating training and validation data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SBR_Net(in_channels_rfv=in_channels_rfv, 
                    in_channels_stack=in_channels_stack,
                    num_blocks=num_blocks).to(device)
    model.apply(kaiming_he_init)

    # Mixed precision floating point arithmetic to speed up training on GPUs
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)

    # criterion = PinballLoss(quantile=0.1) # Default is 0.1
    criterion = torch.nn.BCELoss()
    train_and_validate(run=run, net=model, train_loader=train_loader, val_loader=val_loader,
                    device=device, optimizer=optimizer, scaler=scaler, 
                    lr_scheduler=lr_scheduler, criterion=criterion, num_epochs=num_epochs,
                    use_mixed_precision=use_mixed_precision)
    
    wandb.finish() # End loggin with wandb

    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration of running the program: {duration: .2f} seconds")

if __name__ == "__main__":
    main()