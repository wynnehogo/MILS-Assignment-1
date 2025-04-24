import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsummary import summary

# Dataset
class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        print(f"Loading dataset from {txt_file}...")
        self.data = []
        self.transform = transform
        self.base_path = "D:/C/wyns/0. MILS"
        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.base_path, path)
                if os.path.exists(full_path):
                    self.data.append((full_path, int(label)))
        print(f"Dataset loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms with augmentation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Custom CNN with improvements
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Evaluation
def evaluate(model, dataloader, device):
    print("Evaluating model...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_labels == all_preds).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Evaluation - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1

# Training
def train(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"Validation - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

# Hyperparameter tuning
def hyperparameter_tuning(train_loader, val_loader, device):
    best_model = None
    best_score = 0
    results = []

    learning_rates = [0.001, 0.0005]
    optimizers = [optim.Adam, optim.SGD]

    for lr in learning_rates:
        for opt in optimizers:
            print(f"\nTuning - Optimizer: {opt.__name__}, LR: {lr}")
            model = ImprovedCNN(num_classes=100).to(device)
            optimizer = opt(model.parameters(), lr=lr)
            train(model, train_loader, val_loader, optimizer, device, num_epochs=10)
            acc, prec, rec, f1 = evaluate(model, val_loader, device)

            results.append({
                "Optimizer": opt.__name__,
                "Learning Rate": lr,
                "Val Accuracy": round(acc, 4),
                "Val Precision": round(prec, 4),
                "Val Recall": round(rec, 4),
                "Val F1 Score": round(f1, 4)
            })

            if acc > best_score:
                best_score = acc
                best_model = model

    df = pd.DataFrame(results)
    print("Validation Summary:")
    print(df.to_string(index=False))
    df.to_csv("validation_results.csv", index=False)
    print("Saved validation results to 'validation_results.csv'.")
    return best_model

# Main
if __name__ == "__main__":
    device = torch.device("cpu")

    train_txt = "D:/C/wyns/0. MILS/train.txt"
    val_txt = "D:/C/wyns/0. MILS/val.txt"
    test_txt = "D:/C/wyns/0. MILS/test.txt"

    print("Loading datasets...")
    train_dataset = MiniImageNetDataset(train_txt, transform)
    val_dataset = MiniImageNetDataset(val_txt, transform)
    test_dataset = MiniImageNetDataset(test_txt, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Training with Hyperparameter Tuning...")
    best_model = hyperparameter_tuning(train_loader, val_loader, device)

    print("Testing Best Model...")
    acc, prec, rec, f1 = evaluate(best_model, test_loader, device)
    print(f"Test - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    print("Model Summary:")
    summary(best_model, (3, 64, 64))


