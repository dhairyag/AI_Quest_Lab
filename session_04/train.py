import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Check for CUDA or MPS availability
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Training settings
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, optimizer, and loss function
model = MNISTNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training history
history = {'train_loss': [], 'test_accuracy': []}

def train_epoch(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # Save current training status
            with open('static/training_status.json', 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'progress': 100. * batch_idx / len(train_loader)
                }, f)
    
    avg_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_loss)
    return avg_loss

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    history['test_accuracy'].append(accuracy)
    return test_loss, accuracy

def save_loss_plot():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['test_accuracy'])
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    with open('static/plot.json', 'w') as f:
        json.dump({'plot': plot_data}, f)

def evaluate_random_samples():
    model.eval()
    test_samples = []
    with torch.no_grad():
        # Get 10 random samples
        indices = random.sample(range(len(test_dataset)), 10)
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Convert image to base64 for display
            plt.figure(figsize=(2, 2))
            plt.imshow(image.cpu().squeeze(), cmap='gray')
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            test_samples.append({
                'image': img_data,
                'predicted': pred,
                'actual': label
            })
    
    with open('static/test_samples.json', 'w') as f:
        json.dump(test_samples, f)

def main():
    Path('static').mkdir(exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(epoch)
        test_loss, accuracy = test()
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.2f}%')
        save_loss_plot()
    
    # Save model
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    
    # Evaluate random samples
    evaluate_random_samples()

if __name__ == '__main__':
    main() 