import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', action='store_true', 
    help="train on gpu")
parser.add_argument('-u', '--cpu', action='store_true',
	help="train on cpu")
parser.add_argument('-l', '--linear', action='store_true',
	help="use fully connected layers")
parser.add_argument('-n', '--cnn', action= 'store_true',
	help="use cnn")
parser.add_argument('-t', '--train', action='store_true',
	help="train and test model")
parser.add_argument('-p', '--plot', action='store_true',
	help="plot loss and epoch")
parser.add_argument('-s', '--save', action='store_true',
	help="save model")

args = parser.parse_args()

if args.cuda:
    device = torch.device('cuda')
    print('Using device:', device)
    print('GPU:', torch.cuda.get_device_name(0))
if args.cpu:
	device = torch.device('cpu')
	print("USING CPU")
if args.linear:
	net = FullyConnectedNet().to(device)
	print("//USING FULLY CONNECTED LAYERS//")
if args.cnn:
	net = CnnNet().to(device)
	print("//using cnn//")


print("*"*40, "LOADING DATA", "*"*40)

trainset = torchvision.datasets.MNIST(root='./data', train=True,
										download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
											shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root="./data", train=False,
										download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
											shuffle=True, num_workers=2)
print("="*40, "DATA LOADED", "="*40)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

num_epochs = 13
total_step = len(trainloader)
loss_values = []

def train():
	print("TRAINING")
	for epoch in range(num_epochs):
		running_loss = 0
		for i, (inputs, labels) in enumerate(trainloader):
			if args.linear:
				inputs, labels = inputs.reshape(-1, 28*28).to(device), labels.to(device) 
			if args.cnn:
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
	print("TESTING")
	with torch.no_grad():
		correct = 0
		total = 0
		for inputs, labels in testloader:
			if args.linear:
				inputs, labels = inputs.reshape(-1, 28*28).to(device), labels.to(device)
			if args.cnn:
				inputs, labels = inputs.to(device), labels.to(device)

			output = net(inputs)

			_, predicted = torch.max(output.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print('Accuracy: {} %'.format(100 * correct / total))

if args.train:
	train()
	test()
if args.plot:
	plt.plot(loss_values)
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.show()
if args.save:
	torch.save(net.state_dict(), 'MNIST_model.pt')