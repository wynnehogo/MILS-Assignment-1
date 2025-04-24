# %% Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# %% Dataset
class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, transform=None, channels="RGB"):
        self.data = []
        self.transform = transform
        self.channels = channels
        self.base_path = "D:/C/wyns/0. MILS"
        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.base_path, path)
                if os.path.exists(full_path):
                    self.data.append((full_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        if self.channels in ["R", "G", "B"]:
            channel_idx = {"R": 0, "G": 1, "B": 2}[self.channels]
            image_np = image_np[:, :, channel_idx]
            image = Image.fromarray(image_np, mode='L')
        if self.transform:
            image = self.transform(image)
        return image, label

# %% Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# %% Dynamic Convolution Module
class DynamicConv(nn.Module):
    def __init__(self, max_in_channels, out_channels, kernel_size):
        super(DynamicConv, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.max_in_channels = max_in_channels
        hidden_dim = 128
        self.weight_generator = nn.Sequential(
            nn.Linear(max_in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels * max_in_channels * kernel_size * kernel_size)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        stats = torch.mean(x, dim=[2, 3])
        weight = self.weight_generator(stats)
        weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
        output = [torch.nn.functional.conv2d(x[i].unsqueeze(0), weight[i][:, :c, :, :], padding=self.kernel_size // 2)
                  for i in range(b)]
        return torch.cat(output, dim=0)

# %% Improved Dynamic Model
class DynamicModel(nn.Module):
    def __init__(self, num_classes=100, in_channels=3):
        super(DynamicModel, self).__init__()
        self.block1 = nn.Sequential(
            DynamicConv(max_in_channels=in_channels, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# %% Training & Evaluation Utilities
def compute_metrics(y_true, y_pred):
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images, lbls = images.to(device), lbls.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            preds.append(predicted.cpu().numpy())
            labels.append(lbls.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return compute_metrics(labels, preds)


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=5, print_every=100):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        acc, prec, rec, f1 = evaluate(model, val_loader)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1} completed in {elapsed:.2f}s - Avg Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}\n")


# %% Final Test
def test(model, test_loader):
    acc, prec, rec, f1 = evaluate(model, test_loader)
    print(f"Test: Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

# %% Run
if __name__ == "__main__":
    device = torch.device("cpu")
    channels_list = ['R', 'G', 'B']
    evaluation_results = {}

    for ch in channels_list:
        print(f"\nRunning for channel: {ch}")
        in_channels = 1 if ch in ['R', 'G', 'B'] else 3
        model = DynamicModel(num_classes=100, in_channels=in_channels).to(device)

        train_dataset = MiniImageNetDataset("D:/C/wyns/0. MILS/train.txt", transform, ch)
        val_dataset = MiniImageNetDataset("D:/C/wyns/0. MILS/val.txt", transform, ch)
        test_dataset = MiniImageNetDataset("D:/C/wyns/0. MILS/test.txt", transform, ch)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=5)
        acc, prec, rec, f1 = evaluate(model, val_loader)
        evaluation_results[ch] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

        print("Testing...")
        test(model, test_loader)

    print("\nSummary:")
    for ch, metrics in evaluation_results.items():
        print(f"{ch}: {metrics}")
