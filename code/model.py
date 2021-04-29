
import torch.nn as nn

class SelfDrivingModel(nn.Module):
    """
    This is the model proposed in paper https://arxiv.org/pdf/1604.07316.pdf, except for the dropout layer to reduce overfitting.
    """

    def __init__(self):
        super(SelfDrivingModel, self).__init__()

        # convonlutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.Dropout(0.1)
        )

        # fully connected layers 
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """
        Forward propagation 
        """
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


