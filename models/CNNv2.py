from torch import nn

#More Complex Convolutional Network
class Model(nn.Module):

	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes=2, training_size=0, in_size=40):
        super(Model, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=1)
        size = (in_size - 5 + 2)/1 + 1 
        
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=1)
        size = (size - 5 + 2)/1 + 1 
        
        self.max_pool1 = nn.MaxPool2d(kernel_size = 4, stride = 2, padding=1)
        size = (size - 4 + 2)/2 + 1 
        
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=1)
        size = (size - 5 + 2)/1 + 1 
        
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1)
        size = (size - 5 + 2)/1 + 1 
        
        self.max_pool2 = nn.MaxPool2d(kernel_size = 4, stride = 2)
        size = (size - 4)/2 + 1  
        
        size = 32*(size**2) 
        print(size)
        self.fc1 = nn.Linear(int(size), 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.name = "CNNv2"
        self.training_size = training_size
    
    # Progresses data across layers    
    def forward(self, x):
        self.net = nn.Sequential(
            self.conv_layer1,
            nn.ReLU(),
            self.conv_layer2,
            nn.ReLU(),
            self.max_pool1,
            
            self.conv_layer3,
            nn.ReLU(),
            self.conv_layer4,
            nn.ReLU(),
            self.max_pool2,
            
            nn.Flatten(),
            
            self.fc1,
            self.dropout,
            nn.ReLU(),
            self.fc2,
            #self.dropout,
            self.sigmoid
        )

        return self.net(x)