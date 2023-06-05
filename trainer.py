import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import time


def train_network(net, train_loader, test_loader, num_epochs=100, lr=1e-3, opt="SGD"):

    params = 0
    depth = len(list(net.parameters()))
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        if idx!=depth-2:
            param.requires_grad = False
        print(param)
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

    if opt=="SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.cuda()
    best_acc = 0

    for i in range(num_epochs):
        #print("Epoch: ", i)
        train_loss = train_step(net, optimizer, train_loader)

        if i%10==0:
            test_loss = val_step(net, test_loader)
            train_acc = get_acc(net, train_loader)
            test_acc = get_acc(net, test_loader)

            if train_loss < 1e-15:
                break
            if test_acc > best_acc:
                best_acc = test_acc
                #net.cpu()
                #d = {}
                #d['state_dict'] = net.state_dict()
                #torch.save(d, 'trained_model_best.pth')
                #net.cuda()

            #filters = list(net.parameters())[0].detach().cpu() # 256, 3, 3, 3
            #filters = filters.reshape(len(filters),in_channels,(kernel_size)**2,1) # 256, 3, 9, 1
            #M = torch.mean(filters @ filters.transpose(2,3),dim=0)
            #for c in range(in_channels):
            #    print("channel {}".format(c))
            #    print(M[c])
            #for c in range(in_channels):
            #    print("channel {}".format(c))
            #    print(torch.diagonal(M[c]).reshape((kernel_size,kernel_size)))

            print("Epoch: ", i,
                  "Train Loss: ", train_loss, "Test Loss: ", test_loss,
                  "Train Acc: ", train_acc, "Test Acc: ", test_acc,
                  "Best Test Acc: ", best_acc)




def train_step(net, optimizer, train_loader):
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)
    
    criterion = torch.nn.MSELoss()
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs).cuda())
        target = Variable(targets).cuda()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        loss = criterion(output, target)
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(net, loader):
    net.eval()
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        labels = torch.argmax(output[:, :10], dim=-1)
        target= torch.argmax(target[:, :10], dim=-1)
        count += torch.sum(labels == target).cpu().data.numpy()
    return count / len(loader.dataset) * 100
