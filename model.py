import torch
import torch.nn as nn

class GestureDetection(nn.Module):

    def __init__(self, num_classes):
        super(GestureDetection, self).__init__()

        self.conv_layer1 = self.conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
        self.conv_layer2 = self.conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
        self.conv_layer3 = self.conv_layer(128, 256, (2, 2, 2), (2, 2, 2))

        self.fc3 = nn.Linear(102400, 1024)
        self.act = nn.SELU()
        self.fc4 = nn.Linear(1024, num_classes)

    def conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.Dropout3d(p=0.1),
            nn.SELU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    input_tensor = torch.autograd.Variable(torch.rand(5, 3, 18, 84, 84))
    model = GestureDetection(27) 
    output = model(input_tensor)
    print(output.size())
