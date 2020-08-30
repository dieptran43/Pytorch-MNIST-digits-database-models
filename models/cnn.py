import torch
import torch.nn as nn

class CnnNet(nn.Module):
  def __init__(self):
    super(CnnNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(64*7*7, 128) # it is 64....
    self.fc2 = nn.Linear(128, 10)
    self.dropout = torch.nn.Dropout(p=0.5)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.conv2(x) 
    x = self.relu(x)
    x = self.pool(x) 
    x = x.reshape(x.size(0), -1) 
    x = self.fc1(x) 
    x = self.dropout(x)

    prediction = self.fc2(x)

    return prediction
