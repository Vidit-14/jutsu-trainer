import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from collections import Counter

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
DATA_DIR   = "data"
MODEL_PATH = "jutsu_cnn.pth"
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 0.001

SIGNS = ["rat", "ox", "tiger", "hare", "dragon",
         "snake", "horse", "ram", "monkey", "bird", "dog", "boar"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Using device: {device}")


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════
class HandSignDataset(Dataset):
    def __init__(self, data_dir, signs, transform=None):
        self.samples   = []
        self.transform = transform
        self.signs     = signs

        for label_idx, sign in enumerate(signs):
            folder = os.path.join(data_dir, sign)
            if not os.path.exists(folder):
                print(f"⚠  Missing folder: {folder}")
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png")):
                    self.samples.append((
                        os.path.join(folder, fname),
                        label_idx
                    ))

        print(f"📂 Total images loaded: {len(self.samples)}")
        counts = Counter(s[1] for s in self.samples)
        for idx, sign in enumerate(signs):
            print(f"   {sign:<10} {counts.get(idx, 0)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════════
#  AUGMENTATION
# ═══════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ═══════════════════════════════════════════════════════════════
#  CNN MODEL
# ═══════════════════════════════════════════════════════════════
class JutsuCNN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 → 32
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32 → 16
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16 → 8
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════
def train():
    # Load full dataset first (without transform to split properly)
    full_dataset = HandSignDataset(DATA_DIR, SIGNS, transform=None)

    # Split 80/20
    n_total = len(full_dataset)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    # Apply transforms after split
    train_set.dataset.transform = train_transform
    val_set.dataset.transform   = val_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"\n🏋  Train: {n_train}  |  Val: {n_val}")

    # Model, loss, optimizer, scheduler
    model     = JutsuCNN(num_classes=len(SIGNS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    print(f"\n{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Acc':<12}")
    print("─" * 52)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        total_loss = correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        marker = " ← best" if val_acc > best_val_acc else ""
        print(f"{epoch:<8} {train_loss:<14.4f} {train_acc*100:<14.1f} "
              f"{val_acc*100:<12.1f}{marker}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "signs":       SIGNS,
                "img_size":    IMG_SIZE,
            }, MODEL_PATH)

    print(f"\n✅ Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"💾 Model saved to {MODEL_PATH}")
    print("\nRun main.py next!")

if __name__ == "__main__":
    train()