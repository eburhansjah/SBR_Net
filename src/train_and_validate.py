import torch
import os
import tqdm


def train_and_validate(net, optimizer, device, train_loader, val_loader, criterion, num_epochs):
    if torch.cuda.is_available():
        net.cuda()
    
    root_dir = './runs'
    os.makedirs(root_dir, exist_ok=True)

    model_dir = os.path.join(root_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    best_model_path = os.path.join(model_dir, 'best_ser_model.pth')
    
    best_val_loss = float('inf')
    
    num_batches_train = len(train_loader)
    num_batches_val = len(val_loader)

    ##########
    # Training loop
    ##########            
    for epoch in range(num_epochs):
        running_loss = 0
        correct_pred = 0
        total_pred = 0

        y_true_train = []
        y_pred_train = []

        trainloader_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        net.train()
        for i, data in enumerate(trainloader_iter):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize inputs
            mean_inputs, std_inputs = inputs.mean(), inputs.std()
            inputs = (inputs - mean_inputs) / std_inputs

            # Initializing by zero-ing out gradient
            optimizer.zero_grad()

            # Forward + Backward + Optimization
            forward_output = net(inputs) # pred
            loss = criterion(forward_output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Prediction
            _, pred_indices = torch.max(forward_output, 1)
            label_indices = torch.argmax(labels, dim=1)

            # Counting correct predictions
            correct_pred += (pred_indices == label_indices).sum().item()
            total_pred += pred_indices.shape[0]

            # Print loss and accuracy every specified iterations
            if (i + 1) % 50 == 0:
                current_loss = running_loss / (i + 1)
                current_accuracy = correct_pred / total_pred
                print(f'Epoch: {epoch + 1}, Iteration: {i + 1}, Train Loss: {current_loss:.2f}, Train Accuracy: {current_accuracy:.2f}')

        avg_loss = running_loss / num_batches_train
        accuracy = correct_pred / total_pred
        
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, Accuracy: {accuracy:.2f}')

        ##########
        # Validation loop
        ##########
        running_loss = 0.0
        correct_pred = 0
        total_pred = 0
        y_true_val = []
        y_pred_val = []
        
        with torch.no_grad():
            val_loader_iter = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}', leave=False)
            net.eval()
            for j, data in enumerate(val_loader_iter):
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize inputs
                mean_inputs, std_inputs = inputs.mean(), inputs.std()
                inputs = (inputs - mean_inputs) / std_inputs

                # Forward + Backward + Optimization
                forward_output = net(inputs) # pred
                loss = criterion(forward_output, labels)

                running_loss += loss.item()

                # Prediction
                _, pred_indices = torch.max(forward_output, 1)
                label_indices = torch.argmax(labels, dim=1)

                # Counting correct predictions
                correct_pred += (pred_indices == label_indices).sum().item()
                total_pred += pred_indices.shape[0]

                # Print loss and accuracy every specified iterations
                if (j + 1) % 50 == 0:
                    current_loss = running_loss / (j + 1)
                    current_accuracy = correct_pred / total_pred
                    print(f'Epoch: {epoch + 1}, Iteration: {j + 1}, Val Loss: {current_loss:.2f}, Val Accuracy: {current_accuracy:.2f}')
            

            # Calculating metrics for validation
            avg_loss = running_loss / num_batches_val
            accuracy = correct_pred / total_pred
            print(f'Epoch: {epoch + 1}, Val Loss: {avg_loss:.2f}, Val Accuracy: {accuracy:.2f}')
            
            # Saving the best model
            if avg_loss < best_val_loss:
                print("MODEL UPDATED")
                best_val_loss = avg_loss
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, best_model_path)
            
            # Saving model after each epoch
            epoch_model_path = os.path.join(model_dir, f'see_epoch{epoch + 1}.pth')
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, epoch_model_path)

    print('Finished Training and Validating')

    return net