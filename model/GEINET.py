import torch.nn as nn

GEINet2 = nn.Sequential(
        nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(18),
        nn.ReLU(inplace=True),
        nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(18),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(18, 45, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(45),
        nn.ReLU(inplace=True),
        nn.Conv2d(45, 45, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(45),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.27),
        nn.Linear(45, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 124)
    )