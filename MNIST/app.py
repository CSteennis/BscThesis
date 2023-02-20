
import random as rd

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np


# SRM(spike response model)
# Pj(t)=∑iK ∑t(g)_i∈F_i w_ij * PSP(Δt_ji) + ∑t(f)_j∈Fj η(t−t(f)_j)
# K = len neurons
# i = index sending neuron
# t(g)_i = time spike fired
# d_ji = delay of synapse value
# F_i = sequence of spikes
# w_i_j = weight i->j
# PSP(t) = e(−t/τm)−e(−t/τs)
# Δt_ji = t − t(g)_i − d_ji
# η(t) = −υ e(t/τr) H(t)
# H(t) = 1 : 0 ? t > 0


def import_data():
    file = 'dataset/train-images.idx3-ubyte'
    train_x = idx2numpy.convert_from_file(file).tolist()

    file = 'dataset/train-labels.idx1-ubyte'
    train_y = idx2numpy.convert_from_file(file).tolist()

    return train_x, train_y

# convert pixels to binary representation for input trains
# TODO: find better encode method
# opties:
#  - rate encoding
#  - latency encoding
def gen_spike_train(train_x):
    input = []
    for row in train_x[0]:
        for col in row:
            bin_rep = []
            for bit in bin(col)[2:].zfill(8):
                bin_rep.append(int(bit))
            input.append(bin_rep)

    return input

# weights initialization
def init_weigths(inputs, hiddens, outputs):
    # weights
    w_in_hidden = []
    w_hidden_out = []

    for i in range(inputs):
        tmp = []
        for j in range(hiddens):
            tmp.append(rd.random())
        w_in_hidden.append(tmp)

    for i in range(hiddens):
        tmp = []
        for j in range(outputs):
            tmp.append(rd.random())
        w_hidden_out.append(tmp)

    return w_in_hidden, w_hidden_out

# adapted version of LIF(leaky integrate-and-fire) and SRM
# using 1 hidden layer and 1 output neuron
# TODO: take inhibitionary neuronal spikes into account
# TODO: more realistic spike timing
def run_snn(input, w_in_hidden, w_hidden_out, inputs, hiddens, outputs, v, decay, n_steps):
    print("start")
    # membrane potential
    p_hidden = [.0] * hiddens
    p_out = [.0] * outputs

    in_output = []
    out = []

    # TODO: forloops to function?
    for t in range(0,n_steps):
        # input to hidden layer
        for j in range(hiddens):
            if len(in_output) < hiddens:
                in_output.append([])
            for i in range(inputs):
                # add if spike in train
                p_hidden[j] += w_in_hidden[i][j] * input[i][t]
            # spike
            if p_hidden[j] > v:
                in_output[j].append(1)
                p_hidden[j] = 0
            else:
                in_output[j].append(0)
            # decay potential over time
            p_hidden[j] -= decay
            if p_hidden[j] < 0:
                p_hidden[j] = 0

        # hidden layer to output
        for j in range(outputs):
            if len(out) < outputs:
                out.append([])
            for i in range(hiddens):
                p_out[j] += w_hidden_out[i][j] * in_output[i][t]

            # pot.append(p_out)
            if p_out[j] > v:
                out[j].append(1)
                p_out[j] = 0
            else:
                out[j].append(0)
            # decay potential over time
            p_out[j] -= decay
            if p_out[j] < 0:
                p_out[j] = 0

    print("done")
    # return output spike train, potentials per timestep
    return out

# plot output spike trains
def plot_result(out, outputs, n_steps):
    # TODO: better way of visualising spike train
    y = [i for i in range(0,n_steps)]

    for i in range(outputs):
        plt.plot(y, out[i], label = str(i))
    plt.savefig("out_train.png")

    # for debug purposes
    # for i in range(hiddens):
    #     plt.plot(y, in_output[i])
    # plt.savefig("hidden out train.png")

    # TODO: plot membrane potential

if __name__ == "main":
    # number of time steps
    n_steps = 8 #ms

    # spike threshold
    v = 3 #mV

    # neuron counts
    inputs = 28 * 28
    hiddens = 2
    outputs = 8

    train_x, train_y = import_data()
    input = gen_spike_train(train_x)

    w_in_hidden, w_hidden_out = init_weigths(inputs, hiddens, outputs)

    # potential decay
    decay = 0.01

    out = run_snn(input, w_in_hidden, w_hidden_out, inputs, hiddens, outputs, v, decay, n_steps)

    plot_result(out, outputs, n_steps)

    # TODO: train snn
    # train snn using:
    # - supervised:
    #   - backpropagation, adapted for snn
    #   - Forward-forward
    # - unsupervised:
    #   - STDP
