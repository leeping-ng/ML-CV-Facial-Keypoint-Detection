import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):
    """
    We will be using the NaimishNet Architecture for FKP detection
    https://arxiv.org/pdf/1710.00977.pdf
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # output size: (N - F)/stride + 1
        # input size: (1, 224, 224) while input to NaimishNet is (1, 96, 96)
        self.conv1 = nn.Conv2d(1, 32, 4)    # (32, 221, 221) 
        self.pool1 = nn.MaxPool2d(2, 2)     # (32, 110, 110)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)   # (64, 108, 108)
        self.pool2 = nn.MaxPool2d(2, 2)     # (64, 54, 54)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)  # (128, 53, 53)
        self.pool3 = nn.MaxPool2d(2, 2)     # (128, 26, 26)
        self.drop3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1) # (256, 26, 26)
        self.pool4 = nn.MaxPool2d(2, 2)     # (256, 13, 13)
        self.drop4 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(256*13*13, 1000)
        self.drop5 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(1000, 68*2)

        # use xavier weight initialisation
        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)
        
    def forward(self, x):

        # four convolutional layers
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # three fully connected layers
        x = self.drop5(F.elu(self.fc1(x)))
        x = self.drop6(F.elu(self.fc2(x)))
        x = self.fc3(x)

        return x