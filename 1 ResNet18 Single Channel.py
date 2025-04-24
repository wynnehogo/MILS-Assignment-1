# %% Importing libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# %% Dataset Class
class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, transform=None, channels="RGB"):
        self.data = []
        self.transform = transform
        self.channels = channels
        self.base_path = "D:/C/wyns/0. MILS"

        print(f"Loading dataset from: {txt_file} with channels: {channels}")

        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.base_path, path)
                if os.path.exists(full_path):
                    self.data.append((full_path, int(label)))
                else:
                    print(f"Warning: Image not found: {full_path}")

        print(f"Loaded {len(self.data)} samples from {txt_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (32, 32), (0, 0, 0))
            label = -1

        image_np = np.array(image)

        if self.channels in ["R", "G", "B"]:
            channel_idx = {"R": 0, "G": 1, "B": 2}[self.channels]
            single_channel = image_np[:, :, channel_idx]
            image_np = np.stack([single_channel] * 3, axis=-1)
            image = Image.fromarray(image_np, mode='RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# %% Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# %% Paths
train_txt = "D:/C/wyns/0. MILS/train.txt"
val_txt = "D:/C/wyns/0. MILS/val.txt"
test_txt = "D:/C/wyns/0. MILS/test.txt"

# %% Model
class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18Custom, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# %% Utils
def compute_metrics(y_true, y_pred):
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

def evaluate(model, dataloader):
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
    return compute_metrics(all_labels, all_preds)

def train(model, train_loader, val_loader, optimizer, num_iterations=1):
    criterion = nn.CrossEntropyLoss()
    for iteration in range(num_iterations):
        print(f"\nTraining iteration {iteration+1}/{num_iterations}...")
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

        acc, prec, rec, f1 = evaluate(model, val_loader)
        print(f"Validation - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

def test(model, test_loader):
    print("\nRunning final evaluation on test set...")
    acc, prec, rec, f1 = evaluate(model, test_loader)
    print(f"Test - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# %% Main Execution
if __name__ == "__main__":
    device = torch.device("cpu")
    channels = ["R", "G", "B"]
    evaluation_results = {}

    for ch in channels:
        print(f"\n==== Channel: {ch} ====")
        model = ResNet18Custom(num_classes=100).to(device)

        print("Loading datasets...")
        train_dataset = MiniImageNetDataset(train_txt, transform, ch)
        val_dataset = MiniImageNetDataset(val_txt, transform, ch)
        test_dataset = MiniImageNetDataset(test_txt, transform, ch)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Starting training...")
        train(model, train_loader, val_loader, optimizer, num_iterations=1)

        print("Testing...")
        metrics = test(model, test_loader)
        evaluation_results[ch] = metrics

    print("\nFinal Results Across Channels:")
    for ch, metrics in evaluation_results.items():
        print(f"{ch}: {metrics}")
