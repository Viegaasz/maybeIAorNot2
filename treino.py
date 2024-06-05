import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CNN_v2 import ClassificationModel
from tqdm import tqdm
import csv

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, patience=10):
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    
    with open('epoch_reportV2.csv', 'w', newline='') as csvfile:
        report_writer = csv.writer(csvfile, delimiter=',')
        report_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy'])
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]', unit='batch')
            for inputs, labels in train_progress_bar:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_progress_bar.set_postfix({'train_loss': f'{train_loss/(train_progress_bar.n+1):.4f}'})

            model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0
            valid_progress_bar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]', unit='batch')
            with torch.no_grad():
                for inputs, labels in valid_progress_bar:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    valid_progress_bar.set_postfix({'valid_loss': f'{valid_loss/(valid_progress_bar.n+1):.4f}'})

            accuracy = 100 * correct / total
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), r'C:\Users\lfbar\Downloads\trabalho_IA\trabalho_IA\IA')
            else:
                epochs_no_improve += 1
                print(f'No improvement. Patience: {epochs_no_improve}/{patience}')
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break

            print(f'Accuracy: {accuracy:.2f}%')
            
            report_writer.writerow([epoch+1, train_loss/len(train_loader), valid_loss/len(valid_loader), accuracy])

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = datasets.ImageFolder(r'C:\Users\lfbar\Downloads\trabalho_IA\trabalho_IA\IA', transform=transform)
valid_dataset = datasets.ImageFolder(r'C:\Users\lfbar\Downloads\trabalho_IA\trabalho_IA\IA', transform=transform)

    
numClassesCount = 0
with open('label.txt', 'w') as f:
        for class_name in train_dataset.classes:
            numClassesCount = numClassesCount + 1
            f.write(f'{class_name}\n')
    
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

model = ClassificationModel(num_classes=numClassesCount)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=100000000000000, patience=10)