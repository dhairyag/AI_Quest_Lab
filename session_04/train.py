import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import json
from pathlib import Path
import random
import base64
from io import BytesIO
from tqdm import tqdm
import logging

# Set Matplotlib to use the 'Agg' backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check for CUDA or MPS availability
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
logging.info(f"Using device: {device}")
if device.type == "cuda":
    logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Training settings
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.01

logging.info(f"Batch Size: {BATCH_SIZE}")
logging.info(f"Learning Rate: {LEARNING_RATE}")
logging.info(f"Epochs: {EPOCHS}")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
logging.info("Loading MNIST dataset...")
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
logging.info(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Initialize model, optimizer, and loss function
model = MNISTNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training history
history = {'train_loss': [], 'test_accuracy': []}

def train_epoch(epoch, pbar):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy for this batch
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        current_acc = 100. * correct / total
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'epoch': f'{epoch}/{EPOCHS}',
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })
        
        if batch_idx % 10 == 0:  # Reduced frequency of status updates
            # Save current training status
            with open('static/training_status.json', 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'accuracy': current_acc,
                    'progress': 100. * ((epoch-1) * len(train_loader) + batch_idx) / (EPOCHS * len(train_loader))
                }, f)
    
    avg_loss = total_loss / len(train_loader)
    final_acc = 100. * correct / total
    history['train_loss'].append(avg_loss)
    return avg_loss, final_acc

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
    logging.info("Evaluating random samples...")
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

class ModelTrainer:
    def __init__(self, kernel_config, model_name):
        self.model = MNISTNet(kernel_config).to(device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'kernel_config': kernel_config
        }

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} ({self.model_name})')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            current_acc = 100. * correct / total
            total_loss += loss.item()
            
            # Update status more frequently
            if batch_idx % 5 == 0:
                status = {
                    'model': self.model_name,
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'accuracy': current_acc,
                    'progress': 100. * batch_idx / len(train_loader),
                    'kernel_config': self.history['kernel_config']
                }
                save_training_status(status)
                
        avg_loss = total_loss / len(train_loader)
        self.history['train_loss'].append(avg_loss)
        
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Store the results in history
        self.history['test_accuracy'].append(accuracy)
        
        # Save current test status
        status = {
            'model': self.model_name,
            'test_loss': test_loss,
            'test_accuracy': accuracy
        }
        with open(f'static/test_status_{self.model_name}.json', 'w') as f:
            json.dump(status, f)
            
        return test_loss, accuracy

def save_comparison_plot():
    try:
        # Create figure without GUI
        plt.figure(figsize=(12, 5))
        
        # Training Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(model1_trainer.history['train_loss'], 'b-', label=f'Model 1')
        plt.plot(model2_trainer.history['train_loss'], 'r-', label=f'Model 2')
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Test Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(model1_trainer.history['test_accuracy'], 'b-', label=f'Model 1')
        plt.plot(model2_trainer.history['test_accuracy'], 'r-', label=f'Model 2')
        plt.title('Test Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        with open('static/plot.json', 'w') as f:
            json.dump({'plot': plot_data}, f)
    except Exception as e:
        logging.error(f"Error in save_comparison_plot: {str(e)}")

# Main training loop
def train_models(config1, config2):
    global model1_trainer, model2_trainer
    
    try:
        logging.info(f"Starting training with configurations:")
        logging.info(f"Model 1 kernels: {config1}")
        logging.info(f"Model 2 kernels: {config2}")
        
        model1_trainer = ModelTrainer(config1, "Model 1")
        model2_trainer = ModelTrainer(config2, "Model 2")
        
        for epoch in range(1, EPOCHS + 1):
            logging.info(f"\nEpoch {epoch}/{EPOCHS}")
            
            # Train and test Model 1
            logging.info("Training Model 1...")
            model1_trainer.train_epoch(epoch)
            test_loss1, accuracy1 = model1_trainer.test()
            logging.info(f"Model 1 - Test Loss: {test_loss1:.4f}, Accuracy: {accuracy1:.2f}%")
            
            # Train and test Model 2
            logging.info("Training Model 2...")
            model2_trainer.train_epoch(epoch)
            test_loss2, accuracy2 = model2_trainer.test()
            logging.info(f"Model 2 - Test Loss: {test_loss2:.4f}, Accuracy: {accuracy2:.2f}%")
            
            # Save comparison plots
            save_comparison_plot()
            
            # Save final status for both models
            save_training_status({
                'model': 'Model 1',
                'epoch': epoch,
                'batch': 'completed',
                'loss': test_loss1,
                'accuracy': accuracy1,
                'progress': 100. * epoch / EPOCHS,
                'kernel_config': config1
            })
            
            save_training_status({
                'model': 'Model 2',
                'epoch': epoch,
                'batch': 'completed',
                'loss': test_loss2,
                'accuracy': accuracy2,
                'progress': 100. * epoch / EPOCHS,
                'kernel_config': config2
            })
            
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        # Save error status
        save_training_status({
            'model': 'Model 1',
            'error': str(e)
        })
        save_training_status({
            'model': 'Model 2',
            'error': str(e)
        })

def main():
    Path('static').mkdir(exist_ok=True)
    logging.info("Server ready. Waiting for training configuration from web interface...")

def save_training_status(status):
    """Save training status to a JSON file"""
    status_file = f'static/training_status_{status["model"]}.json'
    with open(status_file, 'w') as f:
        json.dump(status, f)