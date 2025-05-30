
'''
    Data represented as list of 3-tupples (emission, time, cost)
    In the model we have 2 means of probability distributions, lambda and nju,
        which are distributions of how much do people care about emission and reward respectively.
    Given list of pathes, model output the list of rewards which are probability distribution.
    Each reward is in range from 0 to 1 and given total reward R we give each path r_i*R reward.
'''

lmbd    = 0.5
nju     = 0.5
R       = 10
N       = 10
seed    = 42

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AI(nn.Module):
    def __init__(self, lmbd, nju, R, N):
        super().__init__()
        self.lmbd = lmbd
        self.nju = nju
        self.R = R
        self.N = N
        self.model = nn.Sequential(
            nn.Linear(self.N*3, self.N),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

    def train(self, data, epochs):
        '''
            Expected data of shape (batch_size, N)
        '''
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            optimizer.zero_grad()

            r = self.forward(data["train"])
            print(r.shape)

            emission = data["emission"]
            utilities = -self.lmbd * emission + self.R * r * self.nju
            probs = torch.softmax(utilities, dim=1)
            loss = torch.tensor(len(data["test"]))

            for i in range(N):
                loss += probs[:, i] * emission[:, i]

            loss.backward()
            optimizer.step()



def transform(data):
    emissions = []
    train = []
    for (emission, time, cost) in data:
        emissions.append(emission)
        train.append(emission)
        train.append(time)
        train.append(cost)
    return {
        "train": train,
        "emission": emissions,
    }


model = AI(lmbd, nju, R, N)


