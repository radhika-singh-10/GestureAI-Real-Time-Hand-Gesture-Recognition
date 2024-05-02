import torch
import torch.nn as nn
import torch.nn.init as init

# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# https://pytorch.org/docs/stable/nn.init.html
class GestureDetection(nn.Module):
    """
    Convolutional Neural Network for Gesture Detection.
    """

    def __init__(self, num_classes):
        """
        Initialize GestureDetection model.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super(GestureDetection, self).__init__()

        self.conv1 = self._conv_block(3, 128, (1, 2, 2), (1, 2, 2))
        self.conv2 = self._conv_block(128, 256, (2, 2, 2), (2, 2, 2))
        self.conv3 = self._conv_block(256, 256, (2, 2, 2), (2, 2, 2))
        self.conv4 = self._conv_block(256, 256, (2, 2, 2), (2, 2, 2))

        self.fc5 = nn.Linear(12800, 512)
        self.fc5_bn = nn.BatchNorm1d(512)
        self.fc5_activation = nn.LeakyReLU()
        self.fc6 = nn.Linear(512, num_classes)

        self._initialize_weights()


    def _conv_block(self, in_channels, out_channels, pool_size, stride):
        """
        Create a convolutional block.
        """
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(p=0.1),
            nn.LeakyReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.fc5_bn(x)
        x = self.fc5_activation(x)
        x = self.fc6(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)


if __name__ == "__main__":
    input_tensor = torch.autograd.Variable(torch.rand(7, 3, 18, 84, 84))
    model = GestureDetection(7)
    output = model(input_tensor)
    print("Output Tensor Size From This Model:",output.size())

