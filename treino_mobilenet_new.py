# Treino MobileNetV2 com anti-overfitting (augment forte + dropout + label smoothing
# + early stopping + fine-tuning em 2 fases), usando Adam.


import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


# CONFIG
DATA_DIR = r"C:\Data_Science\Aulas\6_Semestre\Deep Learning\Codes_antigos\code_flores_v3\dados"
TRAIN_FOLDER = "flores_train"
VAL_FOLDER = "flores_validation"


IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2


EPOCHS_HEAD = 6        # backbone congelado
EPOCHS_FT = 20         # fine-tuning
LR_HEAD = 3e-3
LR_FT = 3e-4


WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.35
LABEL_SMOOTHING = 0.10


PATIENCE = 7           # early stopping por val_loss
UNFREEZE_BLOCKS = 4    # quantos blocos finais descongelar no fine-tuning


SAVE_PATH = "mobilenetv2_best.pth"



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_path = os.path.join(DATA_DIR, TRAIN_FOLDER)
    val_path = os.path.join(DATA_DIR, VAL_FOLDER)

    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Nao encontrei a pasta de treino: {train_path}")
    if not os.path.isdir(val_path):
        raise FileNotFoundError(f"Nao encontrei a pasta de validacao: {val_path}")

    # Transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Datasets / Loaders
    train_ds = datasets.ImageFolder(train_path, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_path, transform=val_tfms)

    num_classes = len(train_ds.classes)
    print(f"Classes: {num_classes}")
    print("Exemplo classes:", train_ds.classes[:min(10, num_classes)])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Modelo
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Helpers
    def run_epoch(loader, train: bool, optimizer=None):
        model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = criterion(logits, y)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        return total_loss / total, correct / total

    def train_with_early_stopping(epochs, optimizer, scheduler, tag):
        best_wts = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        pat = 0

        for epoch in range(epochs):
            tr_loss, tr_acc = run_epoch(train_loader, train=True, optimizer=optimizer)
            va_loss, va_acc = run_epoch(val_loader, train=False)

            scheduler.step()

            print(
                f"[{tag} {epoch+1:02d}/{epochs}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
            )

            if va_loss < best_val:
                best_val = va_loss
                best_wts = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f"EarlyStopping acionado (patience={PATIENCE}).")
                    break

        model.load_state_dict(best_wts)
        return best_val

    # Fase 1: backbone congelado
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = Adam(model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS_HEAD))
    train_with_early_stopping(EPOCHS_HEAD, optimizer, scheduler, tag="F1")

    # Fase 2: fine-tuning (descongela ultimos blocos)
    if UNFREEZE_BLOCKS > 0:
        for p in model.features[-UNFREEZE_BLOCKS:].parameters():
            p.requires_grad = True

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FT,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS_FT))
    best_val = train_with_early_stopping(EPOCHS_FT, optimizer, scheduler, tag="F2")

    # Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "classes": train_ds.classes,
        "img_size": IMG_SIZE,
        "best_val_loss": best_val,
        "arch": "mobilenet_v2",
        "dropout": DROPOUT_P,
        "label_smoothing": LABEL_SMOOTHING,
    }
    torch.save(ckpt, SAVE_PATH)
    print(f"Checkpoint salvo em: {SAVE_PATH}")



if __name__ == "__main__":
    main()
