from torch.utils.data import Dataset, DataLoader

from snn_net import Net
import torch, torch.nn as nn
import snntorch.functional as sf
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from scipy.signal import savgol_filter
# from plotting import *
import numpy as np

def normalize(x):
    return (x-np.nanmin(x, axis=0))/(np.nanmax(x, axis=0)-np.nanmin(x, axis=0))

class Networks:

    def __init__(self, epochs, inputs, hiddens, outputs, decay, n_steps = 25) -> None:
        self.__params = {}
        self.__params['inputs'] = inputs
        self.__params['hiddens'] = hiddens
        self.__params['outputs'] = outputs
        self.__params['epochs'] = epochs
        self.__params['decay'] = decay
        self.__params['n_steps'] = n_steps

        # self.all_data = all_data
        # self.Ts = Ts

        self._create_nets()

    def set_params(self, **params):
        '''Set params for Networks, usage: `network.set_params(param1 = val1, param2 = val2)`'''
        # Check if param is in parameter dict
        for param, val in params:
            if param in self.__params:
                # edit parameter value
                self.__params[param] = val

        self._create_nets()

    def _create_nets(self):
        self.__rate_snn = Net(self.__params['inputs'],
                            self.__params['hiddens'],
                            self.__params['outputs'],
                            self.__params['decay'],
                            self.__params['n_steps'], enc_type='rate')
        self.__temp_snn = Net(self.__params['inputs'],
                            self.__params['hiddens'],
                            self.__params['outputs'],
                            self.__params['decay'],
                            self.__params['n_steps'], enc_type='latency')
        self.__feed_fwd_net = nn.Sequential(nn.Linear(self.__params['inputs'], self.__params['hiddens']),
                                    nn.ReLU(),
                                    nn.Linear(self.__params['hiddens'], self.__params['outputs']))

    def _test_accuracy(self, data_loader, net, accuracy):
        """ Test the accuracy of the given `net` on the given `data_loader` """
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

    # def predict(self, net, spiking=True):
    #     predictions = []
    #     with torch.no_grad():
    #         for T in self.Ts:
    #             # get all imgs for temp T
    #             x = torch.from_numpy(np.clip(self.all_data['%.3f'%T].astype('float32'),0,1))#['x']
    #             # predict
    #             p = net(x)
    #             # sum spikes over time
    #             if spiking:
    #                 p = p.sum(dim=0)
    #                 p = p.div(p.sum(dim=1, keepdim=True))
    #                 p[p != p] = 0
    #             p = p.nanmean(dim=0).detach().numpy()
    #             predictions.append(p)
    #         x = np.array(predictions)
    #         return normalize(x)

    def _train(self, net: nn.Module, optimizer, loss_fn, accuracy, train_loader, test_loader, epochs, spiking=True):
        '''Training loop for network'''

        acc_hist = []
        loss_hist = []
        test_acc_hist = []
        acc_per_epoch = []

        for epoch in range(epochs):
            # with tqdm(train_loader, unit="batch") as tqepch:
            #     tqepch.set_description(desc=f"Epoch {epoch + 1}")
                # for *_, data, label in tqepch:
            for *_, data, label in train_loader:
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

                # tqepch.set_postfix(loss=loss_val.item(),
                #                 acc=f'{acc * 100:.2f}')

            # accuracy per epoch
            acc_per_epoch.append(acc_hist)

            # accuracy on test set for epoch
            test_acc = self._test_accuracy(test_loader, net, accuracy)
            test_acc_hist.append(test_acc)

            # if epoch % 5 == 0:
            #     prediction = self.predict(net, spiking=spiking)
            #     predictions.append(prediction)

            # print(f'Test accuracy: {test_acc * 100:.2f}%')

        # take the mean of all the epochs
        acc_per_epoch = np.mean(acc_per_epoch, axis=0)

        # smoothing
        acc_per_epoch = savgol_filter(acc_per_epoch, 10, 1)
        loss_hist = savgol_filter(loss_hist, 10, 1)

        # plot
        # plot_accuracy(acc_hist, "Train accuracy")
        # plot_loss(loss_hist, "Train loss")
        # plot_accuracy(test_acc_hist, "Test accuracy", test=True)

        return test_acc_hist, loss_hist, acc_hist

    def train_rate_snn(self, train_loader, test_loader):
        # optimization algoritm
        optimizer = torch.optim.Adam(self.__rate_snn.parameters()) # (NOTE: Adam stond in de tutorial misschien beter algoritme)

        # loss function
        loss_fn = sf.ce_count_loss() # type: ignore

        # accuracy function
        accuracy = sf.accuracy_rate

        return self._train(self.__rate_snn, optimizer, loss_fn, accuracy, train_loader, test_loader,
                           self.__params['epochs']), self.__rate_snn

    def train_temp_snn(self, train_loader, test_loader):
        # optimization algoritm
        optimizer = torch.optim.Adam(self.__temp_snn.parameters()) # (NOTE: Adam stond in de tutorial misschien beter algoritme)

        # loss function
        loss_fn = sf.ce_temporal_loss() # type: ignore

        # accuracy function
        accuracy = sf.accuracy_temporal

        return self._train(self.__temp_snn, optimizer, loss_fn, accuracy, train_loader, test_loader,
                           self.__params['epochs']), self.__temp_snn

    def train_ffn(self, train_loader, test_loader):
        # optimization algoritm
        optimizer = torch.optim.Adam(self.__feed_fwd_net.parameters()) # (NOTE: Adam stond in de tutorial, misschien beter algoritme)

        # loss function
        loss_fn = nn.CrossEntropyLoss()

        # accuracy function
        accuracy = MulticlassAccuracy(num_classes=self.__params['outputs'])

        return self._train(self.__feed_fwd_net, optimizer, loss_fn, accuracy, train_loader, test_loader,
                           self.__params['epochs'], spiking=False), self.__feed_fwd_net
