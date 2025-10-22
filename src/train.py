"""Training script for emotion recognition model."""
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from tqdm import tqdm
import numpy as np

from utils.visualization import (
    log_training_stats, log_epoch_stats, 
    visualize_predictions, log_model_graph,
    log_hyperparameters
)

def create_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, List[str], np.ndarray]:
    """Create train and validation dataloaders with class info."""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Only resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    train_dir = os.path.join(data_dir, "train")
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    classes = full_dataset.classes
    print(f"[INFO] Found {len(classes)} classes: {classes}")
    
    # Compute split sizes
    n_samples = len(full_dataset)
    n_val = int(0.2 * n_samples)  # 20% for validation
    n_train = n_samples - n_val
    
    # Random split with fixed seed for reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Compute class counts for weighting
    class_counts = np.bincount(
        [full_dataset.targets[i] for i in train_dataset.indices],
        minlength=len(classes)
    )
    
    return train_loader, val_loader, classes, class_counts

# ---------- Config ----------
# Dataset
DATA_DIR = Path(r"D:\nam 4\emotion_recognition_project\data")
KAGGLE_DATASET = "ananthu017/emotion-detection-fer"

# Model
BACKBONE = 'efficientnet_b3'  # b0-b7 available
UNFREEZE_BACKBONE = False
MODEL_PATH = r"D:\nam 4\emotion_cnn.pth"

# Training
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 50
LR = 1e-3  # initial learning rate
WEIGHT_DECAY = 1e-4
PATIENCE = 7  # early stopping
GRAD_CLIP = 1.0

# Mixed precision training
USE_AMP = True
USE_ONECYCLE = True  # use one cycle policy

# Output
MODEL_DIR = Path("checkpoints")
LOG_DIR = Path("runs/emotion_train")

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # dataloader workers

# Visualization config
LOG_IMAGES = True  # Log sample predictions
LOG_GRADIENTS = True  # Log gradient distributions
LOG_FREQUENCY = 100  # Log every N batches

# Visualization config
LOG_IMAGES = True  # Log sample predictions
LOG_GRADIENTS = True  # Log gradient distributions
LOG_FREQUENCY = 100  # Log every N batches
# -----------------------------

import random
from torch.utils.data import Subset  # Add missing imports

def download_dataset():
    # chỉ tải nếu thư mục DATA_DIR chưa tồn tại
    if not os.path.exists(DATA_DIR):
        os.makedirs("data", exist_ok=True)
        print("[INFO] Dataset not found. Downloading from Kaggle...")
        os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p data --unzip')
        print("[INFO] Download complete.")
    else:
        print(f"[INFO] Dataset folder exists at: {DATA_DIR}")

def get_transforms(train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(10, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

class EmotionModel(nn.Module):
    def __init__(self, num_classes, backbone='efficientnet_b0', unfreeze_backbone=False):
        super().__init__()
        # select backbone dynamically
        if hasattr(models, backbone):
            self.model = getattr(models, backbone)(weights="IMAGENET1K_V1")
        else:
            # fallback
            self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        # replace classifier head
        # efficientnet classifier layout: classifier[1] is Linear
        try:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        except Exception:
            # generic fallback for unexpected model layouts
            self.model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(self.model.classifier.in_features, num_classes))
        # freeze backbone by default
        for param in self.model.features.parameters():
            param.requires_grad = unfreeze_backbone

    def forward(self, x):
        return self.model(x)


def prepare_data():
    print(f"[INFO] Loading dataset from: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ Dataset folder not found: {DATA_DIR}")

    transform_train = get_transforms(train=True)
    transform_val = get_transforms(train=False)

    # Load dataset
    full = datasets.ImageFolder(DATA_DIR, transform=transform_train)
    classes = full.classes
    print(f"[INFO] Found {len(classes)} classes: {classes}")

    # Stratified split
    targets = np.array(full.targets)
    num_classes = len(classes)
    indices_per_class = [np.where(targets == i)[0] for i in range(num_classes)]
    train_indices, val_indices = [], []
    for idxs in indices_per_class:
        idxs = idxs.tolist()
        random.shuffle(idxs)
        n_val = max(1, int(0.2 * len(idxs)))
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])
    train_ds = Subset(full, train_indices)
    val_ds = Subset(full, val_indices)
    val_ds.dataset.transform = transform_val

    # Class counts for weights
    class_counts = np.bincount(targets, minlength=num_classes)
    # return train indices so caller can build samplers if desired
    return train_ds, val_ds, classes, class_counts, train_indices

def main():
    """Main training function."""
    
    # Create output directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    train_loader, val_loader, classes, class_counts = create_dataloaders(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Create model
    model = EmotionModel(
        num_classes=len(classes),
        backbone=BACKBONE,
        unfreeze_backbone=UNFREEZE_BACKBONE
    ).to(DEVICE)
    print(f"[INFO] Using {BACKBONE} backbone on {DEVICE}")

    # Loss function with class weights
    class_weights = torch.tensor(
        class_counts.sum() / (len(classes) * class_counts),
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch
    )

    # Mixed precision
    scaler = amp.GradScaler() if USE_AMP else None
    
    # TensorBoard
    writer = SummaryWriter(LOG_DIR)
    
    # Training state
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_labels = []
        progress = tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}")
        
        for batch_idx, (imgs, labels) in enumerate(progress):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Mixed precision training
            with amp.autocast(enabled=USE_AMP):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            # Track predictions and metrics
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
            # Save for confusion matrix
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Log to TensorBoard
            if batch_idx % LOG_FREQUENCY == 0:
                # Current batch metrics
                cur_loss = running_loss / (batch_idx + 1)
                cur_acc = 100. * correct / total
                
                # Log metrics
                writer.add_scalar('Loss/train_step', cur_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Accuracy/train_step', cur_acc, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch * len(train_loader) + batch_idx)
                
                # Update progress bar
                progress.set_postfix({
                    'loss': f'{cur_loss:.4f}',
                    'acc': f'{cur_acc:.2f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            if USE_AMP:
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total,
                'lr': f"{current_lr:.2e}"
            })
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        
        # Validation
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            progress = tqdm(val_loader, desc="Validating")
            for batch_idx, (imgs, labels) in enumerate(progress):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                # Track metrics
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                
                # Save for confusion matrix
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Log to TensorBoard
                if batch_idx % LOG_FREQUENCY == 0:
                    cur_loss = val_loss / (batch_idx + 1)
                    cur_acc = 100. * val_correct / val_total
                    
                    writer.add_scalar('Loss/val_step', cur_loss, epoch * len(val_loader) + batch_idx)
                    writer.add_scalar('Accuracy/val_step', cur_acc, epoch * len(val_loader) + batch_idx)
                    
                    progress.set_postfix({
                        'loss': f'{cur_loss:.4f}',
                        'acc': f'{cur_acc:.2f}%'
                    })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        
        # Log metrics
        print(f"\n[E{epoch}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val': val_acc
        }, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'classes': classes,
                'best_acc': best_acc,
                'config': {
                    'backbone': BACKBONE,
                    'batch_size': BATCH_SIZE,
                    'image_size': IMAGE_SIZE
                }
            }
            torch.save(checkpoint, MODEL_DIR / 'model_best.pth')
            print(f"[INFO] ✅ Saved best model (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        # If using ReduceLROnPlateau fallback, step with val_acc
        if not USE_ONECYCLE:
            try:
                scheduler.step(val_acc)
            except Exception:
                pass

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_PATH)
            print(f"[INFO] ✅ Saved best model to {MODEL_PATH} (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[EARLY STOP] No improvement after {PATIENCE} epochs. Stopping early at epoch {epoch}.")
                break

        # Unfreeze backbone if plateau
        # staged unfreeze: unfreeze backbone at epoch 10 for fine-tuning if not improving
        if epoch == 10 and best_acc < 60.0:
            print("[INFO] Unfreezing backbone for fine-tuning...")
            for param in model.model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR/10, weight_decay=1e-4)
            if not USE_ONECYCLE:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            else:
                # recreate OneCycleLR with remaining epochs
                try:
                    remaining_epochs = max(1, EPOCHS - epoch)
                    steps_per_epoch = len(train_loader)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR/10, epochs=remaining_epochs, steps_per_epoch=steps_per_epoch)
                except Exception:
                    pass

    writer.close()
    print(f"[DONE] Training complete. Best val acc: {best_acc:.2f}% at epoch {best_epoch}")

if __name__ == "__main__":
    main()