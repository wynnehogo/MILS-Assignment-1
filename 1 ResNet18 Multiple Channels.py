# %% Imports
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
from collections import defaultdict

# %% Dataset Class (RG, GB, RGB)
class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, transform=None, channels="RGB", max_classes=10, max_images_per_class=50):
        self.data = []
        self.transform = transform
        self.channels = channels.upper()
        self.base_path = "D:/C/wyns/0. MILS"

        seen = defaultdict(int)
        allowed_classes = set(range(max_classes))

        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                label = int(label)
                full_path = os.path.join(self.base_path, path)

                if (label in allowed_classes and 
                    seen[label] < max_images_per_class and 
                    os.path.exists(full_path)):
                    self.data.append((full_path, label))
                    seen[label] += 1

        print(f"[{txt_file}] Loaded {len(self.data)} images for {channels} channels.")

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
        channel_map = {"R": 0, "G": 1, "B": 2}
        selected_indices = [channel_map[c] for c in self.channels if c in channel_map]
        selected_channels = image_np[:, :, selected_indices]

        if selected_channels.shape[2] == 1:
            image_np = np.repeat(selected_channels, 3, axis=2)
        elif selected_channels.shape[2] == 2:
            pad = np.zeros_like(selected_channels[:, :, 0:1])
            image_np = np.concatenate([selected_channels, pad], axis=2)
        else:
            image_np = selected_channels

        image = Image.fromarray(image_np.astype(np.uint8), mode='RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# %% Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# %% Paths
train_txt = "D:/C/wyns/0. MILS/train.txt"
val_txt = "D:/C/wyns/0. MILS/val.txt"
test_txt = "D:/C/wyns/0. MILS/test.txt"

# %% Model
class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=10):
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
    channels_list = ["RB","RG","GB","RGB"]
    evaluation_results = {}

    for ch in channels_list:
        print(f"\n==== Channel: {ch} ====")
        model = ResNet18Custom(num_classes=10).to(device)

        print("Loading datasets...")
        train_dataset = MiniImageNetDataset(train_txt, transform, ch, max_classes=10, max_images_per_class=50)
        val_dataset = MiniImageNetDataset(val_txt, transform, ch, max_classes=10, max_images_per_class=50)
        test_dataset = MiniImageNetDataset(test_txt, transform, ch, max_classes=10, max_images_per_class=50)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Starting training...")
        train(model, train_loader, val_loader, optimizer, num_iterations=1)

        print("Testing...")
        metrics = test(model, test_loader)
        evaluation_results[ch] = metrics

    print("\nFinal Results Across Channels:")
    for ch, metrics in evaluation_results.items():
        print(f"{ch}: {metrics}")
