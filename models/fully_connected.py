import torch
import torch.nn as nn

class FullyConnectedNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 800)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(800, 1600)
		self.fc3 = nn.Linear(1600, 10)


	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)

		return x
