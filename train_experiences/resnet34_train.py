import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resnet import resnet34  
from dataset_new import ChineseCharacterDataset  
import pandas as pd
import os

def train_model(model, train_loader, val_loader, num_classes, batch_size=64, learning_rate=0.01, num_epochs=50, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    print("Model architecture:")
    print(model)

    # Configure the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    print(f"Training parameters: Batch size = {batch_size}, Learning rate = {learning_rate}, Epochs = {num_epochs}")

    # Variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Create directory for saving checkpoints if it doesn't exist
    checkpoint_dir = 'checkpoints/resnet34'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize running loss for the epoch

        # Create tqdm progress bar for training
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU if available

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)  # Compute the loss

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear the gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters

                running_loss += loss.item() * images.size(0)  # Accumulate the running loss

                pbar.set_postfix(loss=loss.item())  # Update the progress bar with the current loss
                pbar.update(1)  # Increment the progress bar

        epoch_loss = running_loss / len(train_loader.dataset)  # Compute the average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Validate the model
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation
            # Create tqdm progress bar for validation
            with tqdm(total=len(val_loader), desc=f'Validation Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    pbar.update(1)

        val_loss /= len(val_loader.dataset)  # Compute the average validation loss
        val_accuracy = 100 * correct / total  # Compute the validation accuracy
        print(f'Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}')
        print(f'Validation Accuracy after Epoch {epoch + 1}: {val_accuracy:.2f}%')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model with epoch number and validation accuracy
            checkpoint_path = os.path.join(checkpoint_dir, f'resnet34_epoch_{epoch + 1}_valacc_{val_accuracy:.2f}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved at {checkpoint_path}!')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    print('Training complete.')

# Main script
if __name__ == '__main__':
    # Training parameters
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 50
    patience = 5  # Number of epochs with no improvement before early stopping

    # Read the labels file using pandas
    labels_df = pd.read_csv('data/952_labels.txt', sep='\s+', header=None, usecols=[0, 1], names=['label', 'character'])
    label_to_char = dict(zip(labels_df['label'], labels_df['character']))
    char_to_label = {v: k for k, v in label_to_char.items()}

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure the size matches the model requirements
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    train_dataset = ChineseCharacterDataset('data/952_train', char_to_label, transform=transform)
    val_dataset = ChineseCharacterDataset('data/952_val', char_to_label, transform=transform)
    test_dataset = ChineseCharacterDataset('data/952_test', char_to_label, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    num_classes = len(set(train_dataset.labels))  # Number of classes from the training dataset
    model = resnet34()  # Initialize ResNet34 model

    # Modify the model's output layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Train the model
    train_model(model, train_loader, val_loader, num_classes, batch_size, learning_rate, num_epochs, patience)
