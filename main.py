import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torchvision.models as models
import time
import wandb
import pprint

from loss import PinballLoss
from src.train_and_validate import train_and_validate
from torch.utils.data import DataLoader, random_split
from torch.profiler import profile, record_function, ProfilerActivity
from data_loader import read_pq_file, TiffDataset
from src.SBR_NET import SBR_Net, kaiming_he_init


def main():
    start_time = time.time()

    # Initializing wanb sweep
    run = wandb.init()
    config = run.config

    print("wandb configurations:")
    pprint.pprint(config)
    
    # Flag on training one or multiple samples
    train_single_sample = config.get("train_single_sample")

    # Flag on whether or not to use mixed precision
    use_mixed_precision = config.get("use_mixed_precision")

    # print("type of flags: ", type(train_single_sample), type(use_mixed_precision)) They're BOOLS

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

    patch_size = config.get("patch_size")

    criterion_name = config.get("criterion")

    quantile = config.get("quantile") # Default is 0.1

#     ##Vascular dataset
#     # input_file_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/training_data/UQ/vasc/15/metadata.pq"
    
    # Non-vascular dataset
    input_file_path = "metadata.pq"

    # Reading .tiff files
    stack_paths, rfv_paths, truth_paths = read_pq_file(input_file_path, one_sample=train_single_sample)

    # Converting .tiff files into tensor
    dataset = TiffDataset(stack_paths=stack_paths, rfv_paths=rfv_paths, truth_paths=truth_paths, patch_size=patch_size)

    if train_single_sample == True:
        print("Training single sample")

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        print("Training more than one samples")

        # Splitting dataset randomly into training and validation (80% and 20% respectively)
        train_sz = round(0.8 * len(dataset))
        val_sz = round(0.2 * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
        
        # Creating training and validation data loaders. batch_size=9 in paper
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

    if criterion_name == "PinballLoss":
        print(f"Criterion used is {criterion_name}")
        print(f"Quantile used is {str(quantile)}")

        fig_title = f"criterion = {criterion_name}, quantile = {str(quantile)}"
        criterion = PinballLoss(quantile)

        train_and_validate(run=run, net=model, train_loader=train_loader, val_loader=val_loader,
                    device=device, optimizer=optimizer, scaler=scaler, 
                    lr_scheduler=lr_scheduler, criterion=criterion, num_epochs=num_epochs,
                    use_mixed_precision=use_mixed_precision, fig_title=fig_title)

    elif criterion_name == "BCELoss":
        print(f"Criterion used is {criterion_name}")

        fig_title = f"criterion = {criterion_name}"
        criterion = torch.nn.BCELoss()

        train_and_validate(run=run, net=model, train_loader=train_loader, val_loader=val_loader,
                    device=device, optimizer=optimizer, scaler=scaler, 
                    lr_scheduler=lr_scheduler, criterion=criterion, num_epochs=num_epochs,
                    use_mixed_precision=use_mixed_precision, fig_title=fig_title)
    else:
        print(f"{criterion_name} not recognized. Please choose either PinballLoss or BCELoss.")
    
    wandb.finish() # End loggin with wandb

    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration of running the program: {duration: .2f} seconds")

if __name__ == "__main__":
    main()
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         main()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))