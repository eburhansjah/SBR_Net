import torch
import os
import tifffile
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def loss_plot_and_save(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig('train_val_loss_plot.png')
    return

def train_and_validate(net, train_loader, val_loader, device, optimizer, scaler, lr_scheduler, criterion, num_epochs):
    if torch.cuda.is_available():
        net.cuda()

    save_output_dir = '/projectnb/tianlabdl/eburhan/SBR_Net/output/'
    os.makedirs(save_output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir='logs')
    ##########
    # Training loop
    ##########            
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)

        net.train()
        train_loss = []
        train_pb = tqdm(enumerate(train_loader), 
                            total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        
        for i, (stack, rfv, truth) in train_pb:
            stack, rfv, truth = stack.to(device), rfv.to(device), truth.to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=True):
                fwd_output = net(rfv, stack)
                loss = criterion(fwd_output, truth)

                '''Calling step after every batch update'''
                loss.backward()
                optimizer.step()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # lr_scheduler.step(epoch + i / len(train_loader))

            train_loss.append(loss.item())
            train_pb.set_postfix({"loss": train_loss / (i + 1)}) # Progress bar with updated current loss value

        lr_scheduler.step()


        ##########
        # Validation loop
        ##########
        net.eval()
        val_loss = []
        val_pb = tqdm(enumerate(val_loader), 
                            total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validating")
        
        with torch.no_grad():
            for i, (stack, rfv, truth) in val_pb:
                stack, rfv, truth = stack.to(device), rfv.to(device), truth.to(device)

                with torch.cuda.amp.autocast(enabled=True):
                    fwd_output = net(rfv, stack)
                    loss = criterion(fwd_output, truth)

                val_loss.append(loss.item())
                val_pb.set_postfix({"loss": val_loss / (i + 1)})
        
        # Logging losses with TensorBoardX
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        
        # Calculating average train and validation loss
        sum_train_loss = sum(train_loss)
        sum_val_loss = sum(val_loss)

        avg_train_loss = sum_train_loss / len(train_loader)
        avg_val_loss = sum_val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Saving model output after each epoch
        output_path = os.path.join(save_output_dir, f'epoch_{epoch + 1}_output.tif')
        tifffile.imwrite(output_path, fwd_output.cpu().numpy())

    writer.close()

    print('Finished Training and Validating')

    # Plotting training and validation loss
    loss_plot_and_save(train_loss, val_loss)

    return net