import torch
from torch import nn
from torchsummary import summary
import yaml

# with open("../configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# NUM_CLASSES = config["model"]["num_classes"]


class SoundClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 128)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5120, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SoundClassifier(num_classes=10).to(device)
    summary(model, (1, 64, 173))
