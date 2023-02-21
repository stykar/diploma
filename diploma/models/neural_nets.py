import torch
import numpy as np
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
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
