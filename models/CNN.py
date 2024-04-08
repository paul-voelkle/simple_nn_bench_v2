from torch import nn

#More Complex Convolutional Network
class Model(nn.Module):

	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes=2, training_size=0):
        super(Model, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1024, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.name = "CNN"
        self.training_size = training_size
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        return out