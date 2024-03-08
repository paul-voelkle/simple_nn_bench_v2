from models import __all__
from torch.utils.data import Dataset

class dataset( Dataset ):
    
    def __init__(self, images, labels):
        self.imgs = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx:int):
        return self.imgs[idx], self.labels[idx]

