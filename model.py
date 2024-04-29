# import torch
# import torch.nn as nn

# class GestureDetection(nn.Module):

#     def __init__(self, num_classes):
#         super(GestureDetection, self).__init__()

#         self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
#         self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
#         self.conv_layer3 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))

#         self.fc3 = nn.Linear(102400, 1024)
#         self.fc3_act = nn.SELU()
#         self.fc4 = nn.Linear(1024, num_classes)

#     def _make_conv_layer(self, in_c, out_c, pool_size, stride):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(out_c),
#             nn.Dropout3d(p=0.1),
#             nn.SELU(),
#             nn.MaxPool3d(pool_size, stride=stride, padding=0)
#         )
#         return conv_layer

#     def forward(self, x):
#         x = self.conv_layer1(x)
#         x = self.conv_layer2(x)
#         x = self.conv_layer3(x)

#         x = x.view(x.size(0), -1)

#         x = self.fc3(x)
#         x = self.fc3_act(x)
#         x = self.fc4(x)
#         return x


# if __name__ == "__main__":
#     input_tensor = torch.autograd.Variable(torch.rand(5, 3, 18, 84, 84))
#     model = GestureDetection(7) 
#     output = model(input_tensor)
#     print(output.size())



# # import torch
# # import torch.nn as nn

# # class GestureDetection(nn.Module):

# #     def __init__(self, num_classes):
# #         super(GestureDetection, self).__init__()

# #         self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
# #         self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
# #         self.conv_layer3 = self._make_conv_layer(
# #             128, 256, (2, 2, 2), (2, 2, 2))
# #         self.conv_layer4 = self._make_conv_layer(
# #             256, 256, (2, 2, 2), (2, 2, 2))

# #         self.fc5 = nn.Linear(12800, 512)
# #         self.fc5_act = nn.ELU()
# #         self.fc6 = nn.Linear(512, num_classes)

# #     def _make_conv_layer(self, in_c, out_c, pool_size, stride):
# #         conv_layer = nn.Sequential(
# #             nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
# #             nn.BatchNorm3d(out_c),
# #             nn.ELU(),
# #             nn.MaxPool3d(pool_size, stride=stride, padding=0)
# #         )
# #         return conv_layer

# #     def forward(self, x):
# #         x = self.conv_layer1(x)
# #         x = self.conv_layer2(x)
# #         x = self.conv_layer3(x)
# #         x = self.conv_layer4(x)

# #         x = x.view(x.size(0), -1)

# #         x = self.fc5(x)
# #         x = self.fc5_act(x)

# #         x = self.fc6(x)
# #         return x


# # if __name__ == "__main__":
# #     input_tensor = torch.autograd.Variable(torch.rand(5, 3, 18, 84, 84))
# #     model = GestureDetection(5) #ConvColumn(27).cuda()
# #     output = model(input_tensor) #model(input_tensor.cuda())
# #     print(output.size())



# import torch
# import torch.nn as nn

# class GestureDetection(nn.Module):

#     def __init__(self, num_classes):
#         super(GestureDetection, self).__init__()

#         self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
#         self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
#         self.conv_layer3 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))
#         self.conv_layer4 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))  # Added convolutional layer

#         self.fc5 = nn.Linear(12800, 512)
#         self.fc5_bn = nn.BatchNorm1d(512)  # Batch normalization layer
#         self.fc5_act = nn.ELU()  # Changed activation function
#         self.fc6 = nn.Linear(512, num_classes)

#     def _make_conv_layer(self, in_c, out_c, pool_size, stride):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(out_c),
#             nn.Dropout3d(p=0.1),
#             nn.ELU(),  # Changed activation function
#             nn.MaxPool3d(pool_size, stride=stride, padding=0)
#         )
#         return conv_layer

#     def forward(self, x):
#         x = self.conv_layer1(x)
#         x = self.conv_layer2(x)
#         x = self.conv_layer3(x)
#         x = self.conv_layer4(x)  # Added convolutional layer

#         x = x.view(x.size(0), -1)

#         x = self.fc5(x)
#         x = self.fc5_bn(x)  # Batch normalization
#         x = self.fc5_act(x)
#         x = self.fc6(x)
#         return x

# if __name__ == "__main__":
#     input_tensor = torch.autograd.Variable(torch.rand(7, 3, 18, 84, 84))
#     model = GestureDetection(27)
#     output = model(input_tensor)
#     print(output.size())




# # New Complex Model

# import torch
# import torch.nn as nn

# class GestureDetection(nn.Module):
#     def __init__(self, num_classes):
#         super(GestureDetection, self).__init__()

#         self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
#         self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
#         self.conv_layer3 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))
#         self.conv_layer4 = self._make_conv_layer(256, 512, (2, 2, 2), (2, 2, 2))
#         self.conv_layer5 = self._make_conv_layer(512, 512, (2, 2, 2), (2, 2, 2), increased_dropout=True)

#         self.fc5 = nn.Linear(2048, 512)  # Adjust input size according to the last conv layer output
#         self.fc5_ln = nn.LayerNorm(512)
#         self.fc5_act = nn.LeakyReLU(negative_slope=0.01)
#         self.fc6 = nn.Linear(512, num_classes)

#     def _make_conv_layer(self, in_c, out_c, pool_size, stride, increased_dropout=False):
#         dropout_rate = 0.3 if increased_dropout else 0.1
#         conv_layer = nn.Sequential(
#             nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(out_c),
#             nn.Dropout3d(p=dropout_rate),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.MaxPool3d(pool_size, stride=stride, padding=0)
#         )
#         return conv_layer

#     def forward(self, x):
#         x = self.conv_layer1(x)
#         x = self.conv_layer2(x)
#         x = self.conv_layer3(x)
#         x = self.conv_layer4(x)
#         x = self.conv_layer5(x)

#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer

#         x = self.fc5(x)
#         x = self.fc5_ln(x)
#         x = self.fc5_act(x)
#         x = self.fc6(x)
#         return x

# if __name__ == "__main__":
#     input_tensor = torch.rand(7, 3, 18, 84, 84)  # Example input tensor
#     model = GestureDetection(7)
#     output = model(input_tensor)
#     print(output.size())




# import torch
# import torch.nn as nn
# import torch
# import torch.nn as nn
# import torch.nn.init as init

# class GestureDetection(nn.Module):
#     def __init__(self, num_classes):
#         super(GestureDetection, self).__init__()

#         self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
#         self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
#         self.conv_layer3 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))
#         self.conv_layer4 = self._make_conv_layer(256, 512, (2, 2, 2), (2, 2, 2))
#         self.conv_layer5 = self._make_conv_layer(512, 512, (2, 2, 2), (2, 2, 2), increased_dropout=True)

#         # Adjusted the input size for BatchNorm1d
#         self.fc5 = nn.Linear(2048, 512)
#         self.fc5_ln = nn.BatchNorm1d(512)  # Use BatchNorm1d for 1D input
#         self.fc5_act = nn.LeakyReLU(negative_slope=0.01)
#         self.fc6 = nn.Linear(512, num_classes)

#         # Initialize weights and biases
#         self._initialize_weights()

#     def _make_conv_layer(self, in_c, out_c, pool_size, stride, increased_dropout=False):
#         dropout_rate = 0.3 if increased_dropout else 0.1
#         conv_layer = nn.Sequential(
#             nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(out_c),
#             nn.Dropout3d(p=dropout_rate),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.MaxPool3d(pool_size, stride=stride, padding=0)
#         )
#         return conv_layer

#     def forward(self, x):
#         x = self.conv_layer1(x)
#         x = self.conv_layer2(x)
#         x = self.conv_layer3(x)
#         x = self.conv_layer4(x)
#         x = self.conv_layer5(x)

#         # Ensure correct input size for fully connected layers
#         x = x.view(x.size(0), -1)

#         x = self.fc5(x)
#         x = self.fc5_ln(x)
#         x = self.fc5_act(x)
#         x = self.fc6(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, 0, 0.01)
#                 init.constant_(m.bias, 0)

# if __name__ == "__main__":
#     input_tensor = torch.rand(7, 3, 18, 84, 84)  # Example input tensor
#     model = GestureDetection(7)
#     output = model(input_tensor)
#     print(output.size())




import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.init as init

class GestureDetection(nn.Module):
    def __init__(self, num_classes):
        super(GestureDetection, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
        self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
        self.conv_layer3 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))
        self.conv_layer4 = self._make_conv_layer(256, 512, (2, 2, 2), (2, 2, 2), increased_dropout=True)
#         self.conv_layer5 = self._make_conv_layer(512, 512, (2, 2, 2), (2, 2, 2), increased_dropout=True)

        # Adjusted the input size for BatchNorm1d
        self.fc5 = nn.Linear(25600, 512)
        self.fc5_ln = nn.BatchNorm1d(512)  # Use BatchNorm1d for 1D input
        self.fc5_act = nn.LeakyReLU(negative_slope=0.01)
        self.fc6 = nn.Linear(512, num_classes)

        # Initialize weights and biases
        self._initialize_weights()

    def _make_conv_layer(self, in_c, out_c, pool_size, stride, increased_dropout=False):
        dropout_rate = 0.2 if increased_dropout else 0.1
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.Dropout3d(p=dropout_rate),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
#         x = self.conv_layer5(x)

        # Ensure correct input size for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc5(x)
        x = self.fc5_ln(x)
        x = self.fc5_act(x)
        x = self.fc6(x)
        return x

    def _initialize_weights(self):
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
    input_tensor = torch.rand(7, 3, 18, 84, 84)  # Example input tensor
    model = GestureDetection(7)
    output = model(input_tensor)
    print(output.size())
