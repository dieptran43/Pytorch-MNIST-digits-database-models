import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset = torchvision.datasets.MNIST(root='./data', train=True,
										download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
											shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root="./data", train=False,
										download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
											shuffle=True, num_workers=2)

test_train = False

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
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

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

num_epochs = 13
total_step = len(trainloader)
loss_values = []

def train():
	for epoch in range(num_epochs):
		running_loss = 0
		for i, (inputs, labels) in enumerate(trainloader):
			inputs, labels = inputs.reshape(-1, 28*28).to(device), labels.to(device)

			optimizer.zero_grad()

			output = net(inputs)
			loss = criterion(output, labels)
			
			loss.backward()
			optimizer.step()

			running_loss =+ loss.item() * inputs.size(0)

			if(i+1) % 100 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

		loss_values.append(running_loss / total_step)

def test():
	with torch.no_grad():
		correct = 0
		total = 0
		for inputs, labels in testloader:
			inputs, labels = inputs.reshape(-1, 28*28).to(device), labels.to(device)

			output = net(inputs)

			_, predicted = torch.max(output.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print('Accuracy: {} %'.format(100 * correct / total))

test_train = True
plot_loss = True
save_model = True

if test_train is True:
	train()
	test()

if plot_loss is True:
	plt.plot(loss_values)
	plt.show()

if save_model is True:
    torch.save(net.state_dict(), 'MNIST_model.pt')