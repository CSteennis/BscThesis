import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.signal import savgol_filter
from snntorch import spikegen


class Net(nn.Module):
    '''Spiking Neural network'''

    def __init__(self, num_inputs, num_hiddens, num_outputs, beta, num_steps, enc_type):
        super().__init__()

        self.enc_type = enc_type  # encryption type e.g. rate, latency

        self.num_inputs = num_inputs  # number of inputs
        self.num_hidden = num_hiddens  # number of hidden neurons
        self.num_outputs = num_outputs  # number of output neurons

        self.num_steps = num_steps  # number of timesteps per sample

        # initialize layers
        # connection input and hidden layer
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.lif1 = snn.Leaky(beta=beta)  # hidden layer

        # connection hidden layer and output
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif2 = snn.Leaky(beta=beta)  # output layer

    def forward(self, data):
        '''Run the network for ``num_steps`` with ``data`` as input. Output spiketrains of outputs'''
        # initialize membrane potentials for hidden and output layer
        mem_hid = self.lif1.init_leaky()
        mem_out = self.lif1.init_leaky()

        spike_out_rec = []

        data = self.gen_spike_trains(data, self.num_steps)

        for i in range(self.num_steps):
            input = self.fc1(data[i])
            spike_hid, mem_hid = self.lif1(input, mem_hid)
            hidden_out = self.fc2(spike_hid)
            spike_out, mem_out = self.lif2(hidden_out, mem_out)

            spike_out_rec.append(spike_out)

        return torch.stack(spike_out_rec)

    def gen_spike_trains(self, data, n_steps) -> torch.Tensor:
        ''' Generate spike train
            In: [num_steps, batch, input_size]
            Out: [num_steps, batch, input_size]
        '''

        spike_data = None

        if self.enc_type == 'rate':
            spike_data = spikegen.rate(data, num_steps=n_steps)
        if self.enc_type == 'latency':
            spike_data = spikegen.latency(data, num_steps=n_steps)
        if spike_data == None:
            spike_data = torch.Tensor(0)
        return spike_data


def test_accuracy(data_loader, net, accuracy):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        data_loader = iter(data_loader)
        for *_, data, targets in data_loader:
            data = data.squeeze().flatten(1)

            spk_rec = net(data)

            acc += accuracy(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


def train(net: nn.Module, optimizer, loss_fn, accuracy, train_loader, test_loader, epochs):
    '''Training loop for snn'''

    acc_hist = []
    loss_hist = []
    test_acc_hist = []
    acc_per_epoch = []

    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tqepch:
            tqepch.set_description(desc=f"Epoch {epoch}")
            for *_, data, label in tqepch:
                # set net to training mode
                net.train()

                data = data.squeeze().flatten(1)

                # do forward pass
                output = net(data)

                # calculate loss value
                loss_val = loss_fn(output, label)
                loss_hist.append(loss_val.item())

                # clear previously stored gradients
                optimizer.zero_grad()

                # calculate the gradients
                loss_val.backward()

                # weight update
                optimizer.step()

                # determine batch accuracy
                acc = accuracy(output, label)
                acc_hist.append(acc)

                tqepch.set_postfix(loss=loss_val.item(),
                                   accuracy=f'{acc * 100:.2f}')

        # accuracy per epoch
        acc_per_epoch.append(acc_hist)

        # accuracy on test set for epoch
        test_acc = test_accuracy(test_loader, net, accuracy)
        test_acc_hist.append(test_acc)

        print(f'Test accuracy: {test_acc * 100:.2f}%')

    # take the mean of all the epochs
    acc_per_epoch = np.mean(acc_per_epoch, axis=0)

    # smoothing
    acc_per_epoch = savgol_filter(acc_per_epoch, 10, 1)
    loss_hist = savgol_filter(loss_hist, 10, 1)

    # plot
    # plot_accuracy(acc_per_epoch, "Train accuracy", 1)
    # plot_loss(loss_hist, "Train loss", 2)
    # plot_accuracy(test_acc_hist, "Test accuracy", 3, test=True)

    # return trained network
    return test_acc_hist
