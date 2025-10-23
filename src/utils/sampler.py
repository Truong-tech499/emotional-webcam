"""Balanced sampling utilities."""
from typing import List

import numpy as np
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """Ensures each batch has balanced number of samples from each class."""
    
    def __init__(self, dataset, indices: List[int], batch_size: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        
        # Get labels for all indices
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)[indices]
        else:
            self.labels = np.array([dataset[i][1] for i in indices])
        
        # Group indices by label
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] 
            for label in np.unique(self.labels)
        }
        
        # Ensure batch size is divisible by number of classes
        self.n_classes = len(self.label_to_indices)
        self.n_samples = batch_size // self.n_classes
        if batch_size % self.n_classes != 0:
            print(f"[WARN] Batch size ({batch_size}) not divisible by number of classes ({self.n_classes})")
            print(f"[WARN] Using {self.n_samples * self.n_classes} samples per batch")
            
    def __iter__(self):
        # Create copy of indices for each class
        self.used_indices = {
            label: indices.copy()
            for label, indices in self.label_to_indices.items()
        }
        
        while True:
            # Check if any class has run out of samples
            if any(len(indices) < self.n_samples 
                  for indices in self.used_indices.values()):
                # Refill indices
                self.used_indices = {
                    label: indices.copy()
                    for label, indices in self.label_to_indices.items()
                }
                
            # Create batch
            batch = []
            for label in self.used_indices:
                # Randomly sample n_samples indices with replacement
                sampled = np.random.choice(
                    self.used_indices[label],
                    size=min(self.n_samples, len(self.used_indices[label])),
                    replace=True
                )
                batch.extend(sampled)
                
                # Remove used indices
                self.used_indices[label] = np.setdiff1d(
                    self.used_indices[label],
                    sampled
                )
                
            # Shuffle batch
            np.random.shuffle(batch)
            yield torch.tensor(batch)
            
    def __len__(self):
        # Return number of complete batches
        return len(self.indices) // (self.n_samples * self.n_classes)