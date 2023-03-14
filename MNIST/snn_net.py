import snntorch as snn
import torch, torch.nn as nn
from tqdm import tqdm
import plotting as plt

class Net(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, beta):
        super().__init__()

        self.num_inputs = num_inputs # number of inputs
        self.num_hidden = num_hiddens # number of hidden neurons
        self.num_outputs = num_outputs # number of output neurons

        # beta = 0.9 # global decay rate for all leaky neurons in layer 1
        # beta2 = torch.rand((num_outputs), dtype = torch.float) # independent decay rate for each leaky neuron in layer 2: [0, 1)

        # initialize layers
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden) # connection input and hidden layer
        self.lif1 = snn.Leaky(beta=beta) # hidden layer
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs) # connection hidden layer and output
        self.lif2 = snn.Leaky(beta=beta) # output layer

    def forward_pass(self, data, num_steps):
        '''Run the network for ``num_steps`` with ``data`` as input. Output spiketrains of outputs'''
        # initialize membrane potentials for hidden and output layer
        mem_hid = self.lif1.init_leaky()
        mem_out = self.lif1.init_leaky()

        spike_hid_rec = []
        mem_hid_rec = []
        spike_out_rec = []
        mem_out_rec = []
        in_cur = []

        for i in range(num_steps):
            input = self.fc1(data[i])
            spike_hid, mem_hid = self.lif1(input, mem_hid)
            hidden_out = self.fc2(spike_hid)
            spike_out, mem_out = self.lif2(hidden_out, mem_out)


            in_cur.append(input)
            spike_hid_rec.append(spike_hid)
            mem_hid_rec.append(mem_hid)
            spike_out_rec.append(spike_out)
            mem_out_rec.append(mem_out)

        # in_cur = torch.stack(in_cur, 0)
        # plt.plot_cur_mem_spk(in_cur[:,0].detach().numpy(), torch.stack(mem_hid_rec)[:,0].detach().numpy(), torch.stack(spike_hid_rec)[:,0])

        # plt.plot_snn_spikes(data, torch.stack(spike_hid_rec), torch.stack(spike_out_rec), "Fully Connected Spiking Neural Network", num_steps)

        # plt.anim_plot(torch.stack(spike_out_rec))

        return torch.stack(spike_out_rec)


