import torch
import torch.nn as nn
import torch.nn.functional as F
import layers as l
import torch.optim as optim
from data import train_data, test_data, test_data_aff, test_data_aff_subset, test_data_hom, test_data_hom_subset
import numpy as np
import sampling as s
from lie import LieGroup as lie

device = 'cuda:0'

torch.cuda.empty_cache()

group = 'homography'
G = lie(group, device)

if group == 'affine':
    lie_scales = [0.168, 0.168, 0.841, 0.168, 0.168, 0.841]
elif group == 'homography':
    lie_scales = [0.15, 0.35, 0.5, 0.35, 0.15, 0.5, 0.15, 0.15]

width = 128 # width used in experiments

embedding_dim = 10
output_dims1 = [20, width]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        fl_feature_dims = [1, 40, 40]
        scales = [1, 1]
        fl_MLP_dims = [20, 20]
        self.first_layer = l.FirstLayer(group, fl_feature_dims, embedding_dim, fl_MLP_dims, [1,20], scales, device)

        hl_MLP_dims = [embedding_dim * G.dim, width]
        self.second_layer = l.FilterMLP(embedding_dim, hl_MLP_dims, output_dims1, device)

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

        pooling_input_dim = width
        pooling_output_dim = 10
        self.pooling_layer = l.PoolingLayer(pooling_input_dim, pooling_output_dim, device)

    def forward(self, data, v):
        z = self.first_layer(data, v[-1])
        z = F.silu(z)
        z = (1/v[0].shape[0])*torch.sum(torch.mul(z.unsqueeze(-1), self.second_layer(v[0])), dim=(-2, -3))
        z = F.silu(self.fc1(z)) + z
        z = F.silu(self.fc2(z)) + z
        z = self.pooling_layer(z)
        return z


net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-8)
criterion = nn.CrossEntropyLoss()

epochs = 150
train_acc = np.zeros(epochs)
test_acc = np.zeros(epochs)
test_g_acc = np.zeros(epochs)
generate = s.generate_haar_PI_parallel(group, 100, 100, 1, lie_scales, device)

train_batch = 60
test_batch = 290

trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch, shuffle=False, num_workers=2)

if group == 'affine':
    g_testloader = torch.utils.data.DataLoader(test_data_aff, test_batch, shuffle=False, num_workers=2)
    g_testloader_subset = torch.utils.data.DataLoader(test_data_aff_subset, test_batch, shuffle=False, num_workers=2)
elif group == 'homography':
    g_testloader = torch.utils.data.DataLoader(test_data_hom, test_batch, shuffle=False, num_workers=2)
    g_testloader_subset = torch.utils.data.DataLoader(test_data_hom_subset, test_batch, shuffle=False, num_workers=2)

for e in range(epochs):
    correct = 0
    total = 0

    print("Epoch: ", e)

    for data in trainloader:
            
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        v = generate.generate()
        outputs = net(inputs, v)

        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    train_acc[e] = correct / total
    print("Training accuracy: {}".format(train_acc[e]))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            v = generate.generate()

            outputs = net(images, v)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc[e] = correct / total

        correct = 0
        total = 0
            
        for data in g_testloader_subset:
            images, labels = data[0].to(device), data[1].to(device)
            v = generate.generate()

            outputs = net(images, v)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_g_acc[e] = correct / total

    print("GNIST test accuracy on first 10K images: {}".format(test_acc[e]))
    print("MNIST test accuracy: {}".format(test_g_acc[e]))

with torch.no_grad():

    correct = 0
    total = 0

    for data in g_testloader:
        images, labels = data[0].to(device), data[1].to(device)
        v = generate.generate()

        outputs = net(images, v)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_g_acc_final = correct / total
    
print("Final GNIST test accuracy: {}".format(test_g_acc_final))