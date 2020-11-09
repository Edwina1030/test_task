from torch import nn


class build_model(nn.Module):
    def __init__(self):
        super(build_model, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(16)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(stride=2, kernel_size=2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )

        self.fc = nn.Sequential(nn.Linear(512, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 10)  # 10 type fonts
                                )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


print(build_model())
