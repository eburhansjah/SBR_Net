import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import yaml
import time
import tifffile 
import pandas as pd
from loss import PinballLoss
import train_and_validate

from yaml import load
from torch.utils.data import DataLoader, random_split
from data_loader import read_pq_file, TiffDataset
from src.SBR_NET import SBR_Net, kaiming_he_init

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

if __name__ == "__main__":
    # start_time = time.time()

    # Reading values from config file
    stream = open("config.yaml", 'r')
    docs = yaml.safe_load(stream)

    if docs:
        # Model parameters
        batch_size = docs.get("batch_size")
        in_channels_rfv = docs.get("in_channels_rfv")
        in_channels_stack = docs.get("in_channels_stack")
        num_blocks = docs.get("num_blocks")

        # Optimizer parameters
        learning_rate = docs.get("learning_rate")
        weight_decay = docs.get("weight_decay")

        # Scheduler parameters
        T_max = docs.get("T_max")
        eta_min = docs.get("eta_min")
        last_epoch = docs.get("last_epoch")

        num_epochs = docs.get("num_epochs")

    input_file_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/training_data/UQ/vasc/15/metadata.pq"
    # df = pd.read_parquet(input_file_path)
    # print(df.head())

    # Reading .tiff files
    stack_paths, rfv_paths, truth_paths = read_pq_file(input_file_path)

    # Converting .tiff files into tensor
    dataset = TiffDataset(stack_paths=stack_paths, rfv_paths=rfv_paths, truth_paths=truth_paths)

    # Splitting dataset randomly into training and validation (80% and 20% respectively)
    train_sz = round(0.8 * len(dataset))
    val_sz = round(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
    
    # Creating training and validation data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
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

    criterion = PinballLoss(quantile=0.1) # Default is 0.1

    train_and_validate(net=model, train_loader=train_loader, val_loader=val_loader,
                    device=device, optimizer=optimizer, scaler=scaler, 
                    lr_scheduler=lr_scheduler, criterion=criterion, num_epochs=num_epochs)
    
    # output = model(rfv_input, stack_input)
    # print(output.shape) # Should be: Bx24x224x224

    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"Duration of running the program: {duration: .2f} seconds")