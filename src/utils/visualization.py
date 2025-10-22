"""Visualization utilities for TensorBoard."""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def log_training_stats(writer, model, optimizer, epoch, batch_idx, 
                    loss, acc, phase='train', log_gradients=False):
    """Log detailed training statistics to TensorBoard"""
    # Basic metrics
    writer.add_scalar(f'Loss/{phase}', loss, epoch * batch_idx)
    writer.add_scalar(f'Accuracy/{phase}', acc, epoch * batch_idx)
    
    # Learning rate
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f'Learning_Rate/group_{i}', 
                         param_group['lr'], epoch * batch_idx)
    
    if log_gradients and phase == 'train':
        # Gradient & weight norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradients/total_norm', total_norm, epoch * batch_idx)

        # Layer-wise gradients & weights
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', 
                                   param.grad, epoch * batch_idx)
                writer.add_histogram(f'Weights/{name}', 
                                   param.data, epoch * batch_idx)

def log_epoch_stats(writer, model, epoch, train_loss, val_loss,
                   train_acc, val_acc, classes, all_labels, all_preds):
    """Log epoch-level statistics and visualizations"""
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    writer.add_figure('Confusion_Matrix/val', fig, epoch)
    plt.close(fig)
    
    # Per-class accuracy
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    for i, (cls, acc) in enumerate(zip(classes, acc_per_class)):
        writer.add_scalar(f'Accuracy_per_class/{cls}', acc, epoch)
    
    # Memory usage if CUDA available
    if torch.cuda.is_available():
        writer.add_scalar('Memory/GPU_allocated', 
                         torch.cuda.memory_allocated()/1024/1024, epoch)
        writer.add_scalar('Memory/GPU_cached',
                         torch.cuda.memory_reserved()/1024/1024, epoch)

def visualize_predictions(writer, model, val_loader, classes, device, epoch):
    """Log sample predictions with images"""
    model.eval()
    imgs, labels = next(iter(val_loader))
    imgs = imgs[:8].to(device)  # Get first 8 images
    
    with torch.no_grad():
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu()
    
    # Create grid of images with predictions
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, (img, label, pred) in enumerate(zip(imgs, labels[:8], preds)):
        img = img.cpu().permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = img.clip(0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'True: {classes[label]}\nPred: {classes[pred]}',
                          color='green' if label == pred else 'red')
        axes[idx].axis('off')
    
    plt.tight_layout()
    writer.add_figure('Sample_Predictions', fig, epoch)
    plt.close(fig)

def log_model_graph(writer, model, device):
    """Log model architecture graph"""
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)

def log_hyperparameters(writer, config, metrics):
    """Log hyperparameters and final metrics"""
    writer.add_hparams(
        config,
        metrics,
        run_name='hparams'
    )