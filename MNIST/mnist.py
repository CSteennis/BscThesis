
import matplotlib.pyplot as plt
import snntorch.functional as sf
from snntorch import spikegen
from snntorch import utils
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
from tqdm import tqdm
import numpy as np

import plotting as splt

from snn_net import Net

matplotlib.use("TkAgg")

def import_data():
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST("MNIST/dataset/", train=True, download=True, transform=transform)

    return mnist_train

# rate encode pixels to binary representation for input trains
def gen_spike_trains(data, n_steps):
    ''' Generate spike train
        In: [num_steps, batch, input_size]
        Out: [num_steps, batch, input_size]
    '''
    spike_data = spikegen.rate(data.flatten(1), num_steps=n_steps)
    return spike_data

# TODO: take inhibitionary neuronal spikes into account?
def train_snn(net:Net, optimizer:torch.optim.Adam, loss_fn:sf.mse_count_loss, train_set, n_steps):
    '''Training loop for snn'''
    print("start")

    train_loader = DataLoader(train_set, batch_size=128)

    epochs = 1
    acc_hist = []

    for epoch in range(epochs):
        for i, (data, label) in enumerate(tqdm(iter(train_loader))):
            # convert input to spike trains
            input = gen_spike_trains(data.squeeze(), n_steps)

            # set net to training mode
            net.train()

            # do forward pass
            output = net.forward_pass(input, n_steps)

            loss_val = loss_fn(output, label)

            # clear previously stored gradients
            optimizer.zero_grad()

            # calculate the gradients
            loss_val.backward()

            # weight update
            optimizer.step()

            # print(i)

            if (i % 25) == 0:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

                # check accuracy on a single batch
                acc = sf.accuracy_rate(output, label)
                acc_hist.append(acc)
                print(f"Accuracy: {acc * 100:.2f}%\n")

            # if i == 100:
            #     break

    plot_accuracy(acc_hist, "train accuracy")

    print("done")
    # return output spike train, potentials per timestep
    return net

def plot_accuracy(acc_hist, title):
    print(acc_hist)
    fig = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.title(title)
    plt.xlabel("Measurement")
    plt.ylabel("Accuracy")
    plt.savefig("out/"+title+".png")

if __name__ == "__main__":
    # number of time steps
    n_steps = 200 #ms

    # neuron counts
    inputs = 28 * 28
    hiddens = 200
    outputs = 10

    train_x = import_data()

    # potential decay
    decay = 0.9

    # initialize net
    net = Net(inputs, hiddens, outputs, decay)

    # optimalizatie algoritme
    optimizer = torch.optim.Adam(net.parameters()) # (NOTE: Adam stond in de tutorial wellicht beter algo)

    # loss function
    loss_fn = sf.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2) # type: ignore

    train_snn(net, optimizer, loss_fn, train_x, n_steps)
