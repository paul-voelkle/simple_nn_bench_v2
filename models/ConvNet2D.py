from torch import nn

#Simple Convolutional Network
class Model(nn.Module):

    def __init__(self, training_size=None, in_ch=1, hdn_ch=4, out_dim=2, img_sz=40, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hdn_ch, kernel_size, bias=True, stride=1, padding=2)
        self.conv2 = nn.Conv2d(hdn_ch, 1, kernel_size, bias=True, stride=1, padding=2)
        self.max = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(int(img_sz*img_sz/16), out_dim, bias=True)
        self.name = "ConvNet2D"
        self.training_size = training_size
    
    def forward(self, x):
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.max,
            nn.ReLU(),
            self.conv2,
            self.max,
            self.flatten,
            self.out,
            nn.Sigmoid()
        )
        return self.net(x)