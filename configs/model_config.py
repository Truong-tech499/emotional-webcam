"""Model configuration for emotion recognition."""
from typing import Dict, Any

def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return {
        # Model architecture
        'backbone': 'resnet18',
        'pretrained': True,
        'num_classes': 7,
        
        # Input configuration
        'input_size': (224, 224),
        'channels': 3,
        
        # Model parameters
        'dropout_rate': 0.5,
        'feature_extract': True,  # Only update the reshaped layer params
        
        # Paths
        'checkpoint_dir': 'checkpoints/emotion_cnn',
        'best_model_path': 'checkpoints/emotion_cnn/best.pth',
        'latest_model_path': 'checkpoints/emotion_cnn/latest.pth'
    }