from torch import nn

#Fully Connected Network
class Model(nn.Module):
    
    def __init__(self, training_size=None, img_sz=40, out_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(img_sz*img_sz, out_dim, bias=True)
        self.name = "FullyConnected"
        self.training_size = training_size
    
    def forward(self, x):
        self.net = nn.Sequential(
            nn.Flatten(),
            self.linear1,
            nn.Softmax()
        )
        return self.net(x)