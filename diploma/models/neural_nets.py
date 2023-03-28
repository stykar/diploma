import torch
import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """Class that implements basic utilities"""

    def calculate_params(self) -> int:
        """Calculate model's trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerPerceptron(BaseModel):
    """
    Class that implements simple multilayer perceptron
    """

    def __init__(
        self,
        hidden_layers: list[int],
        input_dim: int,
        output_dim: int,
        dropout_prob: float = 0
    ) -> None:
        """Initiliazes model"""
        super(MultiLayerPerceptron, self).__init__()
        in_features = [input_dim] + hidden_layers
        out_features = hidden_layers + [output_dim]
        layers = []
        for in_feature, out_feature in zip(in_features[:-1], out_features[:-1]):
            layers.append(nn.Linear(in_features=in_feature, out_features=out_feature))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features=in_features[-1], out_features=out_features[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"Forward pass of the model"""
        output = self.hidden_layers(x)
        return self.output_layer(output)

    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the prediction like the other sklearn models
        :param x: numpy array of shape n_samples x n_features
        :returns : numpy array of shape n_samples x n_labels
        """
        x = torch.Tensor(x)
        probs = torch.nn.functional.softmax(self.forward(x=x), dim=-1)
        return probs.numpy()

class ConvolutionalNetwork(BaseModel):
    """
    Class that implements convolutinal network
    """

    def __init__(
        self,
        output_dim: int,
        dropout_prob: float = 0.2
    ) -> None:
        """Initiliazes model"""
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
            # this outputs 64 features
        )
        self.hidden_linear_layers = nn.Sequential(
            nn.Dropout(dropout_prob),
            #nn.LazyLinear(out_features=32)
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"Forward pass of the model"""
        x = x.unsqueeze(-2)
        conv_output = self.conv_layers(x)
        #print(conv_output.shape)
        return self.hidden_linear_layers(conv_output)

    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the prediction like the other sklearn models
        :param x: numpy array of shape n_samples x n_features
        :returns : numpy array of shape n_samples x n_labels
        """
        x = torch.Tensor(x)
        probs = torch.nn.functional.softmax(self.forward(x=x), dim=-1)
        return probs.numpy()