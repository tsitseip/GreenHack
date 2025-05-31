
'''
    Data represented as list of 3-tupples (emission, time, cost)
    In the model we have 2 means of probability distributions, lambda and nju,
        which are distributions of how much do people care about emission and reward respectively.
    Given list of pathes, model output the list of rewards which are probability distribution.
    Each reward is in range from 0 to 1 and given total reward R we give each path r_i*R reward.
'''

lmbd        = 0.5
nju         = 0.9
R           = 100.0
N           = 5
seed        = 42
batch_size  = 5

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
            nn.Linear(self.N*3, self.N*20),
            nn.ReLU(),
            nn.Linear(self.N*20, self.N*40),
            nn.ReLU(),
            nn.Linear(self.N*40, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.N),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def calc_probs(self, emission, r):
        utilities = -self.lmbd * emission + self.R * r * self.nju
        probs = torch.softmax(utilities, dim=2)
        return probs

    def train_model(self, data, epochs):
        '''
            Expected data of shape (batch, N)
        '''
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0
            for batch_idx in range(len(data["train"])):
                batch_train = data["train"][batch_idx]  # shape (batch_size, N*3)
                batch_emission = data["emission"][batch_idx]  # shape (batch_size, N)
                batch_time = data["time"][batch_idx]  # shape (batch_size, N)
                r = self.forward(batch_train)
                utility = self.lmbd * batch_emission - self.R * r * self.nju + batch_time
                utility = torch.softmax(utility, dim=-1)
                loss = -torch.sum(utility * batch_emission, dim=1).mean()
                loss.backward()
                total_loss += loss.item()
            num = len(data["train"])
            # print(epoch, num)
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num}")

    def predict(self, test):
        data = test["train"]
        emission = test["emission"]

        input_tensor = torch.tensor(data, dtype=torch.float32)

        r = self.forward(input_tensor)
        # print(r.shape, r)

        util = self.lmbd * emission - self.R * r * self.nju + test["time"]
        probs = torch.softmax(util, dim=-1)
        loss = torch.sum(probs * emission, dim=1).mean()

        return probs, loss
def transform(data, batch_size, seed):
    generator = np.random.RandomState(seed)

    emissions = [[] for _ in range(len(data) // batch_size)]
    times = [[] for _ in range(len(data) // batch_size)]
    price = [[] for _ in range(len(data) // batch_size)]
    train = [[] for _ in range(len(data) // batch_size)]

    for batch_i in range(len(data) // batch_size):
        permutation = generator.permutation(len(data))
        batch_id = permutation[batch_i * batch_size:(batch_i + 1) * batch_size]
        batch = [data[i] for i in batch_id]
        for test in batch:
            emissions[batch_i].append([])
            times[batch_i].append([])
            price[batch_i].append([])
            train[batch_i].append([])
            np.random.shuffle(test)
            # print(test)
            # print(len(test))
            for (emission, time, cost) in test:
                emissions[batch_i][-1].append(emission)
                times[batch_i][-1].append(time)
                price[batch_i][-1].append(cost)
                train[batch_i][-1].append(emission)
                train[batch_i][-1].append(time)
                train[batch_i][-1].append(cost)
    return {
        "train": torch.Tensor(train),
        "emission": torch.Tensor(emissions),
        "time": torch.Tensor(times),
        "cost": torch.Tensor(price),
    }

def train_m():
    import pickle
    with open('train_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)


    with open("test_dataset.pkl", "rb") as f:
        test = pickle.load(f)

    # print(dataset[:10])
    # print(test[:10])
    # dataset = [
    #     [(2.5, 3.3, 1.6), (5.6, 4.1, 1.1), (0.5, 0.3, 0.1)],
    #     [(6.1, 8.5, 10.6), (5.6, 4.1, 0.2), (0.9, 0.1, 0.1)],
    #     [(4.4, 3.3, 2.2), (8.5, 4.3, 0.0), (0.1, 0.6, 100)]
    # ]

    data = transform(dataset, batch_size, seed)
    test_data = transform(test, 1, seed)

    # print(data["train"].shape)

    model = AI(lmbd, nju, R, N)
    model.train_model(data, 100)


    print(model.predict(test_data))

    torch.save(model.state_dict(), 'model')


    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 60)
    models = [AI(lmbd, i, R, N) for i in x]
    for model in models:
        model.train_model(data, 100)
    y = [model.predict(test_data)[1].item() for model in models]


    plt.plot(x, y, label="Loss vs nju")
    plt.title("Effect of nju on Loss")
    plt.xlabel("nju")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot.png")
    plt.show()

def helper(model, data):
    data_t = transform([data], 1, seed)

    pred = model.predict(data_t)
    ans = []
    for i in pred[0][0][0]:
        ans.append(i.item()*R)
    return ans

#train_m()
