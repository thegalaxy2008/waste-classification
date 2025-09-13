import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super(TinyVGG, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], hidden_units, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_units * (input_shape[1] // 4) * (input_shape[2] // 4), output_shape)
        )
    def forward(self, x):
        x = self.model(x)
        return x
