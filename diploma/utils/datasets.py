import torch
import numpy as np
from torch.utils.data import Dataset


class PharmaDataset(Dataset):
    """Class that implements Pytorch dataset for our problem"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray
        ) -> None:
        """Initiliazes object"""
        super(PharmaDataset).__init__()
        # Convert to ints so we don't have issue with indexing
        labels = labels.astype(np.int16) - 1
        self.x = features
        self.y = np.zeros(shape=(labels.shape[0], 2))
        # Save labels with one-hot encoding
        self.y[np.arange(labels.size), labels] = 1

    def __len__(self):
        """Get dataset length"""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return selected item"""
        return torch.Tensor(self.x[index]), torch.Tensor(self.y[index])