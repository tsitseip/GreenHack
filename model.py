
'''
    Data represented as list of 3-tupples (emission, time, cost)
    In the model we have 2 means of probability distributions, lambda and nju,
        which are distributions of how much do people care about emission and reward respectively.
    Given list of pathes, model output the list of rewards which are probability distribution.
    Each reward is in range from 0 to 1 and given total reward R we give each path r_i*R reward.
'''

lmbd        = 0.5
nju         = 0.5
R           = 10.0
N           = 3
seed        = 42
batch_size  = 1

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
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Linear(self.N*3, self.N),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

    def calc_probs(self, emission, r):
        utilities = -self.lmbd * emission + self.R * r * self.nju
        probs = torch.softmax(utilities, dim=2)
        return probs

    def train(self, data, epochs):
        '''
            Expected data of shape (batch, N)
        '''
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # print(data["train"].shape)
            r = self.forward(data["train"])
            # print(r.shape)

            emission = data["emission"]
            # print(emission.shape)
            probs = self.calc_probs(emission, r)
            loss = torch.zeros([data["train"].shape[0], data["train"].shape[1]])

            for i in range(self.N):
                # print(loss.shape, i, probs[:, :, i], emission[:, :, i], probs[:, :, i] * emission[:, :, i], (probs[:, :, i] * emission[:, :, i]).shape)
                loss += probs[:, :, i] * emission[:, :, i]

            total_loss = loss.sum()

            total_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item()}")

    def predict(self, test):
        emissions = torch.Tensor([emission for (emission, time, cost) in test])

        data = []
        for (emission, time, cost) in test:
            # print(emission, time, cost)
            data.extend([emission, time, cost])

        input_tensor = torch.tensor([data], dtype=torch.float32)

        r = self.forward(input_tensor)
        # print(r.shape, r)

        probs = self.calc_probs(emissions, r.unsqueeze(0))

        loss = torch.tensor(probs * emissions).sum()

        return probs, loss


def transform(data, batch_size, seed):
    generator = np.random.RandomState(seed)
    permutation = generator.permutation(len(data))

    emissions = [[] for _ in range(len(data) // batch_size)]
    train = [[] for _ in range(len(data) // batch_size)]

    for batch_i in range(len(data) // batch_size):
        batch_id = permutation[batch_i * batch_size:(batch_i + 1) * batch_size]
        batch = [data[i] for i in batch_id]
        for test in batch:
            emissions[batch_i].append([])
            train[batch_i].append([])
            for (emission, time, cost) in test:
                emissions[batch_i][-1].append(emission)
                train[batch_i][-1].append(emission)
                train[batch_i][-1].append(time)
                train[batch_i][-1].append(cost)
    return {
        "train": torch.Tensor(train),
        "emission": torch.Tensor(emissions),
    }


dataset = [
    [(2.5, 3.3, 1.6), (5.6, 4.1, 1.1), (0.5, 0.3, 0.1)],
    [(6.1, 8.5, 10.6), (5.6, 4.1, 0.2), (0.9, 0.1, 0.1)],
    [(4.4, 3.3, 2.2), (8.5, 4.3, 0.0), (0.1, 0.6, 100)]
]

data = transform(dataset, batch_size, seed)


model = AI(lmbd, nju, R, N)
model.train(data, 100)

print(model.predict([(4, 5, 1), (6, 5, 4), (3, 4, 5)]))
