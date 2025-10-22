"""CNN model for emotion recognition."""
import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    """CNN model for emotion classification."""
    
    def __init__(self, num_classes: int = 7):
        super().__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify final layer for emotion classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)