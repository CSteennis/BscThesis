import torch, numpy as np
from torch.utils.data import Dataset

from Ising.IsingData import generate_Ising_configurations

import json

class ISING(Dataset):

    def __init__(self, L, train, Ts, train_Ts, avg=False, gen=True, cache=True):
        super().__init__()
        self.train = train

        self.Ts = Ts

        self.L = L

        self.cache = cache

        # generate data if non existing
        self.all_data = self._generate_Ising_data(1000, Ts)

        raw, train_all, test_all = self._get_training_data(train_Ts, avg=avg)
        self.T, self.data, self.targets = train_all if self.train else test_all

    def _get_avg_raw(self, data):
        raw_x = []
        # avg over adjacent pixels
        for x in data:
            norm_x = []
            for i in range(len(x)):
                avg_list = [x[i]]
                if i-1 > 0:
                    avg_list.append(x[i-1]) # left
                if i+1 < self.L:
                    avg_list.append(x[i+1]) # right
                if i-self.L > 0:
                    avg_list.append(x[i-self.L]) # up
                if i+self.L < self.L:
                    avg_list.append(x[i+self.L]) # down
                norm = np.array(avg_list).mean()
                norm_x.append(norm)
            raw_x.append(norm_x)
        raw_x = np.array(raw_x)

        return raw_x

    def _get_training_data(self, Ts, Tc=2.7, train_fraction=0.8, avg=False):
        '''Shuffle and preprocces the data into a train and test batch'''
        # Lists to store the raw data
        raw_T = []
        raw_x = []
        raw_y = []

        for T in Ts:
            raw_x.append(self.all_data['%.3f'%(T)])
            n = len(self.all_data['%.3f'%(T)])
            label = 1 if T < Tc else 0
            raw_y.append(np.array([label] * n))
            raw_T.append(np.array([T]*n))


        raw_T = np.concatenate(raw_T, dtype=np.float32)
        raw_x = np.concatenate(raw_x, axis=0, dtype=np.float32)
        raw_y = np.concatenate(raw_y, axis=0, dtype=np.longlong)

        # Shuffle
        indices = np.random.permutation(len(raw_x))
        all_T = raw_T[indices]
        all_x = raw_x[indices]
        all_y = raw_y[indices]

        # Split into train and test sets
        train_split = int(train_fraction * len(all_x))
        train_T = torch.from_numpy(all_T[:train_split])
        if avg:
            train_x = torch.from_numpy(self._get_avg_raw(np.clip(all_x[:train_split], 0, 1)))
        else:
            train_x = torch.from_numpy(np.clip(all_x[:train_split], 0, 1))
        train_y = torch.from_numpy(all_y[:train_split])
        test_T = torch.from_numpy(all_T[train_split:])
        test_x = torch.from_numpy(np.clip(all_x[train_split:], 0, 1))
        test_y = torch.from_numpy(all_y[train_split:])

        return [raw_T, raw_x, raw_y], [train_T, train_x, train_y], [test_T, test_x, test_y]

    def _generate_Ising_data(self, numSamplesPerT, Ts) -> dict:
        '''If data is present in cache folder, load the data else generate and store the data'''
        try:
            if not self.cache:
                raise
            with open(f'cache/{self.L}-{numSamplesPerT}.json', 'r') as file:
                self.all_data = {key: np.array(val) for key, val in json.load(file).items()}
        except:
            self.all_data = generate_Ising_configurations(self.L, numSamplesPerT, Ts)
            with open(f'cache/{self.L}-{numSamplesPerT}.json', 'w') as file:
                json.dump({key: val.tolist() for key, val in self.all_data.items()}, file)
        return self.all_data

    def get_all_data(self):
        return self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.T[index], self.data[index], self.targets[index]