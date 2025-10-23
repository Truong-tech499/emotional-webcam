"""Training configuration for emotion recognition."""
from typing import Dict, Any

def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return {
        # Training parameters
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        
        # Optimizer settings
        'optimizer': 'adam',
        'momentum': 0.9,
        'beta1': 0.9,
        'beta2': 0.999,
        
        # Learning rate scheduler
        'scheduler': 'cosine',
        'min_lr': 1e-6,
        'warmup_epochs': 3,
        
        # Data augmentation
        'augmentation': {
            'horizontal_flip': True,
            'random_rotation': 10,
            'color_jitter': 0.1
        },
        
        # Paths
        'data_dir': 'data',
        'train_dir': 'data/train',
        'val_dir': 'data/val',
        'test_dir': 'data/test',
        'log_dir': 'runs/emotion_train',
        
        # Validation settings
        'val_frequency': 1,  # Validate every N epochs
        'save_frequency': 5,  # Save checkpoint every N epochs
        
        # Early stopping
        'early_stopping': {
            'patience': 5,
            'min_delta': 0.001
        }
    }