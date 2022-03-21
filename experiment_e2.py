import torch
import torch.nn as nn
import torch.optim as optim
from data import train_data, test_data, test_data_aff, test_data_aff_subset, test_data_hom, test_data_hom_subset
import numpy as np
from e2sfcnn import E2SFCNN as E2

device = 'cuda:0'

torch.cuda.empty_cache()

net = E2(1, 10, 16, 5).to(device)
print("model loaded")

optimizer = optim.Adam(net.parameters(), lr=0.015, betas=(0.9, 0.999), weight_decay=1e-8)
scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.CrossEntropyLoss()

epochs = 40
train_acc = np.zeros(epochs)

train_batch = 64
test_batch = 300

trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch, shuffle=False, num_workers=2)

aff_testloader = torch.utils.data.DataLoader(test_data_aff, test_batch, shuffle=False, num_workers=2)
hom_testloader = torch.utils.data.DataLoader(test_data_hom, test_batch, shuffle=False, num_workers=2)

for e in range(epochs):
    correct = 0
    total = 0

    print("Epoch: ", e)

    for data in trainloader:
            
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if e > 14:
        scheduler1.step()
        
    train_acc[e] = correct / total
    print("Training accuracy: {}".format(train_acc[e]))

torch.save(net.state_dict(), 'e2.pth')

with torch.no_grad():

    correct = 0
    total = 0

    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = correct / total

    print("Final test accuracy: {}".format(test_acc))

    correct = 0
    total = 0

    for data in aff_testloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_aff_acc_final = correct / total

    print("Final affNIST test accuracy: {}".format(test_aff_acc_final))

    correct = 0
    total = 0

    for data in hom_testloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_hom_acc_final = correct / total
    
    print("Final homNIST test accuracy: {}".format(test_hom_acc_final))
