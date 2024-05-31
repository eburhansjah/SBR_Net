import torch
import os
from tqdm import tqdm


def train_and_validate(net, train_loader, val_loader, device, optimizer, scaler, lr_scheduler, criterion, num_epochs):
    if torch.cuda.is_available():
        net.cuda()

    ##########
    # Training loop
    ##########            
    for epoch in range(num_epochs):
        net.train()
        train_loss = []
        train_pb = tqdm(enumerate(train_loader), 
                            total=len(train_loader), dec=f"Epoch {epoch + 1}/{num_epochs} - Training")
        
        for i, (stack, rfv, truth) in train_pb:
            stack, rfv, truth = stack.to(device), rfv.to(device), truth.to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=True):
                fwd_output = net(rfv, stack)
                loss = criterion(fwd_output, truth)

                '''Calling step after every batch update'''
                loss.backward()
                optimizer.step()
                lr_scheduler.step(epoch + i / len(train_loader)) # at end of each epoch??

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())
            train_pb.set_postfix({"loss": train_loss / (i + 1)}) # Progress bar with updated current loss value

        ##########
        # Validation loop
        ##########
        net.eval()
        val_loss = []
        val_pb = tqdm(enumerate(val_loader), 
                            total=len(val_loader), dec=f"Epoch {epoch + 1}/{num_epochs} - Validating")
        
        with torch.no_grad():
            for i, (stack, rfv, truth) in val_pb:
                stack, rfv, truth = stack.to(device), rfv.to(device), truth.to(device)

                with torch.cuda.amp.autocast(enabled=True):
                    fwd_output = net(rfv, stack)
                    loss = criterion(fwd_output, truth)

                val_loss.append(loss.item())
                val_pb.set_postfix({"loss": val_loss / (i + 1)})
        
        # Calculating average train and validation loss
        sum_train_loss = sum(train_loss)
        sum_val_loss = sum(val_loss)

        avg_train_loss = sum_train_loss / len(train_loader)
        avg_val_loss = sum_val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print('Finished Training and Validating')

    return net