from torchvision import transforms
import torch
from data_loader import DogsVsCatsDataset
import spikingjelly.clock_driven as cd
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, encoding

from torch.cuda.amp import GradScaler, autocast
import time


train_ds = "dogs-vs-cats/train"
test_ds = "dogs-vs-cats/test1"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

dataset_rgb = DogsVsCatsDataset(train_ds, color_space='rgb', transform=transform)
loader_rgb = torch.utils.data.DataLoader(dataset_rgb, batch_size=8, shuffle=True)

test_rgb = DogsVsCatsDataset(test_ds, color_space='rgb', transform=transform)
test_loader = torch.utils.data.DataLoader(test_rgb, batch_size=8, shuffle=True)

#dataset_yuv = DogsVsCatsDataset(train_ds, color_space='yuv', transform=transform)
#loader_yuv = torch.utils.data.DataLoader(dataset_yuv, batch_size=32, shuffle=True)

# Set parameters
start_epoch = 0
epochs = 100  # adjust as needed
lr = 0.01  # learning rate
T = 100  # number of time steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

# Initialize the LIF neuron parameters
v_reset = 0.0
v_threshold = 1.0
tau = 100.0

class CatAndDogConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.lif1 = neuron.LIFNode()
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.lif2 = neuron.LIFNode()
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.lif3 = neuron.LIFNode()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.lif4 = neuron.LIFNode()
        
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.lif5 = neuron.LIFNode()
        
        self.fc3 = nn.Linear(in_features=50, out_features=2)
        self.lif6 = neuron.LIFNode()

    def forward(self, X):

        X = self.conv1(X)
        X = F.avg_pool2d(X, 2)
        X = self.lif1(X)
        
        X = self.conv2(X)
        X = F.avg_pool2d(X, 2)
        X = self.lif2(X)

        X = self.conv3(X)
        X = F.avg_pool2d(X, 2)
        X = self.lif3(X)

        X = X.view(X.shape[0], -1)
        X = self.fc1(X)
        X = self.lif4(X)
        
        X = self.fc2(X)
        X = self.lif5(X)
        
        X = self.fc3(X)
        X = self.lif6(X)

        return X


# Initialize the network
net = CatAndDogConvNet()

# Use Adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Use PoissonEncoder
encoder = encoding.PoissonEncoder()

# Training loop
for epoch in range(start_epoch, epochs):
    start_time = time.time()
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for img, label in loader_rgb:  # adjust your data loader
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        label_onehot = F.one_hot(label, 2).float()

        if scaler is not None:
            with autocast():
                out_fr = 0.
                for t in range(T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / T
                loss = F.mse_loss(out_fr, label_onehot)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = 0.
            for t in range(T):
                encoded_img = encoder(img)
                out_fr += net(encoded_img)
            out_fr = out_fr / T
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        cd.functional.reset_net(net)

torch.save(net.state_dict(), 'RGB.pt')