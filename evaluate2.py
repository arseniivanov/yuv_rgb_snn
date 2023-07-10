from torchvision import transforms
import torch
from data_loader import DogsVsCatsDataset
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import snntorch.functional as SF 
import numpy as np
from snntorch import surrogate, utils

from torch.cuda.amp import GradScaler, autocast
import time


train_ds = "dogs-vs-cats/train"
test_ds = "dogs-vs-cats/test1"

transform = transforms.Compose([
    transforms.Resize((224, 224))
])

dataset_rgb = DogsVsCatsDataset(train_ds, color_space='rgb', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset_rgb, batch_size=64, shuffle=True)

test_rgb = DogsVsCatsDataset(test_ds, color_space='rgb', transform=transform)
test_loader = torch.utils.data.DataLoader(test_rgb, batch_size=64, shuffle=True)

#dataset_yuv = DogsVsCatsDataset(train_ds, color_space='yuv', transform=transform)
#loader_yuv = torch.utils.data.DataLoader(dataset_yuv, batch_size=32, shuffle=True)

# Set parameters
start_epoch = 0
epochs = 10  # adjust as needed
lr = 0.1  # learning rate
T = 50  # number of time steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

# Initialize the LIF neuron parameters
v_reset = 0.0
v_threshold = 1.0
tau = 100.0

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

class CatAndDogConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.bnfc1 = nn.BatchNorm1d(num_features=500)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.bnfc2 = nn.BatchNorm1d(num_features=50)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc3 = nn.Linear(in_features=50, out_features=2)    
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, X):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif3.init_leaky()
        mem5 = self.lif3.init_leaky()
        mem6 = self.lif3.init_leaky()

        X = self.conv1(X)
        X = self.bn1(X)

        X, mem1 = self.lif1(X, mem1)
        X = F.avg_pool2d(X, 2)
        
        X = self.conv2(X)
        X = self.bn2(X)
        X, mem2 = self.lif2(X, mem2)
        X = F.avg_pool2d(X, 2)

        X = self.conv3(X)
        X = self.bn3(X)
        X, mem3 = self.lif3(X, mem3)
        X = F.avg_pool2d(X, 2)

        X = X.view(X.shape[0], -1)
        X = self.fc1(X)
        X = self.bnfc1(X)
        X, mem4 = self.lif4(X, mem4)
        
        X = self.fc2(X)
        X = self.bnfc2(X)
        X, mem5 = self.lif5(X, mem5)
        
        X = self.fc3(X)

        return X, mem5

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)
      #Add loss to output
      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total
# Initialize the network
net = CatAndDogConvNet()
net = net.to(device)

# Use Adam optimizer
loss_fn = SF.ce_rate_loss()

# Training loop
num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0
dtype = torch.float
num_steps = 25
batch_size = 128

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):

    # Training loop
    for data, targets in iter(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)

        # initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        if counter % 2 == 0:
          with torch.no_grad():
              net.eval()

              # Test set forward pass
              test_acc = batch_accuracy(test_loader, net, num_steps)
              
              print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
              test_acc_hist.append(test_acc.item())

        counter += 1

torch.save(net.state_dict(), 'RGB.pt')