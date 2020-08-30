import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("LOADING DATA...")
trainset = torchvision.datasets.MNIST(root='./data', train=True,
										download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
											shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root="./data", train=False,
										download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
											shuffle=True, num_workers=2)
print("DATA LOADED")

fully_connected = True
cnn = False

if fully_connected:
	net = FullyConnectedNet().to(device)
	print("Using fully connected")
if cnn:
	net = CnnNet().to(device)
	print("Using cnn")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

num_epochs = 13
total_step = len(trainloader)
loss_values = []

def train():
	print("starting train process")
	for epoch in range(num_epochs):
		running_loss = 0
		for i, (inputs, labels) in enumerate(trainloader):
			if fully_connected:
				inputs, labels = inputs.reshape(-1, 28*28).to(device), labels.to(device) 
			if cnn:
				inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			output= net(inputs)
			loss = criterion(output, labels)
			
			loss.backward()
			optimizer.step()

			running_loss =+ loss.item() * inputs.size(0)

			if(i+1) % 100 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

		loss_values.append(running_loss / total_step)

def test():
	print("starting test process")
	with torch.no_grad():
		correct = 0
		total = 0
		for inputs, labels in testloader:
			if fully_connected:
				inputs, labels = inputs.reshape(-1, 28*28).to(device), labels.to(device)
			if cnn:
				inputs, labels = inputs.to(device), labels.to(device)

			output = net(inputs)

			_, predicted = torch.max(output.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print('Accuracy: {} %'.format(100 * correct / total))

test_train = True
plot_loss = True
save_model = False

if test_train is True:
	train()
	test()

if plot_loss is True:
	plt.plot(loss_values)
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.show()

if save_model is True:
	 torch.save(net.state_dict(), 'MNIST_model.pt')
