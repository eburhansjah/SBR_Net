import os
import torch
import wandb
import matplotlib.pyplot as plt

from tqdm import tqdm


def loss_plot_and_save(run, train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.grid(True)
    plt.legend()
    plt.savefig('train_val_loss_plot.png')

    run.log({"Visualization on training and validation losses": plt})

    plt.close()

    return

'''Fn. for comparing model output and ground truth'''
def compare_output_and_gt(run, gt_tensor, out_tensor, epoch, fig_info):
    gt_tensor = gt_tensor.cpu().detach()
    out_tensor = out_tensor.cpu().detach()

    # Removing extra dimension (batch size)
    gt_tensor = gt_tensor.squeeze(0)
    out_tensor = out_tensor.squeeze(0)

    # Using max. intensity projection (mip) to visualize images with multiple channels
    gt_mip = gt_tensor.max(dim=0).values
    out_mip = out_tensor.max(dim=0).values


    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_mip.numpy())
    plt.colorbar()
    plt.title("Ground truth image")

    plt.subplot(1, 2, 2)
    plt.imshow(out_mip.numpy())
    plt.colorbar()
    plt.title("Model output image")

    plt.suptitle(f"MIP images of gt and model output at epoch: {epoch + 1} with {fig_info}")

    run.log({"Visualization on model output and gt": plt})

    plt.close()

    return


def train_and_validate(run, net, train_loader, val_loader, device, optimizer, 
                       scaler, lr_scheduler, criterion, num_epochs,
                       use_mixed_precision, fig_title):
    if torch.cuda.is_available():
        net.cuda()

    save_output_dir = '/projectnb/tianlabdl/eburhan/SBR_Net/output/'
    os.makedirs(save_output_dir, exist_ok=True)


    ##########
    # Training loop
    ##########
    train_loss = []
    val_loss = []            
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)

        net.train()

        train_pb = tqdm(enumerate(train_loader), 
                            total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        
        epoch_train_loss = 0
        for _, (stack, rfv, truth) in train_pb:
            stack, rfv, truth = stack.to(device), rfv.to(device), truth.to(device)

            print(f"Size of stack: {stack.shape}, size of rfv: {rfv.shape}, size of truth: {truth.shape}")
            """Size of stack: torch.Size([1, 9, 512, 512]), size of rfv: torch.Size([1, 24, 512, 512]), 
            size of truth: torch.Size([1, 24, 512, 512])"""
            optimizer.zero_grad()
            
            if use_mixed_precision == True:
                print('Mixed precision is used during training.')
                
                with torch.cuda.amp.autocast(enabled=True):
                    fwd_output = net(rfv, stack)
                    loss = criterion(fwd_output, truth)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                print('No mixed precision is used during training.')

                fwd_output = net(rfv, stack)
                loss = criterion(fwd_output, truth)
                loss.backward()
                optimizer.step()

            # lr_scheduler.step(epoch + i / len(train_loader))

            epoch_train_loss += loss.item()

            # Progress bar with updated current training loss value
            train_pb.set_postfix({"Current training loss": loss.item()})

        lr_scheduler.step()


        ##########
        # Validation loop
        ##########
        net.eval()
        epoch_val_loss = 0
        val_pb = tqdm(enumerate(val_loader), 
                            total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validating")
        
        with torch.no_grad():
            for _, (stack, rfv, truth) in val_pb:
                stack, rfv, truth = stack.to(device), rfv.to(device), truth.to(device)

                if use_mixed_precision == True:
                    print('Mixed precision is used during validation.')

                    with torch.cuda.amp.autocast(enabled=True):
                        fwd_output = net(rfv, stack)
                        loss = criterion(fwd_output, truth)
                else:
                    print('No Mixed precision is used during validation.')
                    
                    fwd_output = net(rfv, stack)
                    loss = criterion(fwd_output, truth)

                epoch_val_loss += loss.item()

                # Progress bar with updated current validation loss value
                val_pb.set_postfix({"Current validation loss": loss.item()})
        
        # Calculating average train and validation loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Logging avg. losses with wandb
        run.log({'avg_train_loss' : avg_train_loss,
                 'avg_val_loss' : avg_val_loss})
        
    # Logging images with wandb
    compare_output_and_gt(run=run, gt_tensor=truth, out_tensor=fwd_output, epoch=epoch, fig_info=fig_title)

        # # Saving model output after each epoch
        # output_path = os.path.join(save_output_dir, f'epoch_{epoch + 1}_output.tif')
        # tifffile.imwrite(output_path, fwd_output.cpu().numpy())

    print('Finished Training and Validating with values logged in wandb!')

    # Plotting training and validation loss
    loss_plot_and_save(run, train_loss, val_loss)

    return net