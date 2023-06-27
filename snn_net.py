import snntorch as snn
import torch
import torch.nn as nn
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

    def forward(self, data) -> torch.Tensor:
        '''Run the network for ``num_steps`` with ``data`` as input. Output spiketrains of outputs'''
        # initialize membrane potentials for hidden and output layer
        mem_hid = self.lif1.init_leaky()
        mem_out = self.lif2.init_leaky()

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
