import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsummary import summary

# %% Dataset
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

# %% Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# %% ResNet34 Baseline
class ResNet34Baseline(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet34Baseline, self).__init__()
        print("Initializing ResNet34 Baseline...")
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# %% Evaluate Function
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

# %% Train Function
def train(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
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
        print(f"Epoch {epoch + 1}/{num_epochs} completed")
        acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"Validation - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

# %% Main Execution
if __name__ == "__main__":
    device = torch.device("cpu")  

    # Dataset paths
    train_txt = "D:/C/wyns/0. MILS/train.txt"
    val_txt = "D:/C/wyns/0. MILS/val.txt"
    test_txt = "D:/C/wyns/0. MILS/test.txt"

    # Dataset loaders
    print("Loading datasets...")
    train_dataset = MiniImageNetDataset(train_txt, transform)
    val_dataset = MiniImageNetDataset(val_txt, transform)
    test_dataset = MiniImageNetDataset(test_txt, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train ResNet34 Baseline
    print("Training ResNet34 Baseline...")
    baseline_model = ResNet34Baseline(num_classes=100).to(device)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    train(baseline_model, train_loader, val_loader, optimizer, device, num_epochs=10)

    # Test ResNet34
    print("Testing ResNet34...")
    acc, prec, rec, f1 = evaluate(baseline_model, test_loader, device)
    print(f"Test - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # Model Summary
    print("\nModel Summary for ResNet34:")
    summary(baseline_model, (3, 224, 224))
