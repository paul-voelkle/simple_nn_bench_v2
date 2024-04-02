from torch import nn

#More Complex Convolutional Network
class Model(nn.Module):

    def __init__(self, training_size=None, in_ch=1, hdn_ch=4, out_dim=2, img_sz=40, kernel_size=4):
        super().__init__()
        
        self.padd = nn.ZeroPad2d((1,2,1,2))
        
        self.conv1 = nn.Conv2d(1, 128, kernel_size, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size, bias=True)
        
        self.max = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(int(img_sz*img_sz*4),256, bias=True)
        self.linear2 = nn.Linear(256,256, bias=True)
        self.out = nn.Linear(256, out_dim, bias=True)
        
        self.dropout = nn.Dropout2d(0.2) 
        
        self.name = "CNN"
        self.training_size = training_size
    
    def forward(self, x):
        self.net = nn.Sequential(
            self.padd,
            self.conv1,
            nn.ReLU(),
            self.padd,
            self.conv2,
            nn.ReLU(),
            self.max,
            
            self.padd,
            self.conv3,
            nn.ReLU(),
            self.padd,
            self.conv4,
            nn.ReLU(),
            self.max,
            
            self.flatten,
            self.linear1,
            self.dropout,
            nn.ReLU(),
            self.linear2,
            self.dropout,
            nn.ReLU(),
            self.out,
            nn.ReLU(),
            nn.Sigmoid()
        )
        return self.net(x)