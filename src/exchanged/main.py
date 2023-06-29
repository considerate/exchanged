import torch
import math
import numpy as np
from random import random

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.start = torch.nn.Parameter(
            torch.randn(40)/math.sqrt(40),
            requires_grad=True,
        )
        self.linear = torch.nn.Linear(40, 40)
        self.linear2 = torch.nn.Linear(40, 20)
        self.linear3 = torch.nn.Linear(20, 10)
        self.linear4 = torch.nn.Linear(10, 2)

    def forward(self):
        x = self.start.data
        x = self.linear(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.linear4(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x

def text_barchart(arr, low=0.0, high=1.0, rounding = round):
    levels = "▁▂▃▄▅▆▇█"

    def get_level(x):
        t = (x - low) / (high - low)
        if math.isnan(t):
            t = 0
        i = max(0, min(7, rounding(t * 8)))
        return levels[i]

    chars = [get_level(x) for x in arr]
    return "".join(chars)


def main():
    steps = 10000
    runs = 10
    final = []
    bias = 0.25
    debug = False
    for run in range(runs):
        model = Network()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        model_2 = Network()
        optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=1e-2, momentum=0.9)
        model_2.load_state_dict(model.state_dict())
        optimizer_2.load_state_dict(optimizer.state_dict())
        if run == 0:
            print(optimizer)
            print(optimizer_2)
        log_prob_sequence: list[float] = []
        probabilities = []
        flips = [ random () <= bias for _ in range(steps)]
        for step, flip in enumerate(flips):
            optimizer.zero_grad()
            log_probs = model()
            probs = torch.exp(log_probs)
            probabilities.append(probs)
            if debug and step % (len(flips) // 100) == 0:
                print(f"{probs[0].item(): 3.3f} {probs[1].item(): 3.3f}")
            if flip:
                index = 0
            else:
                index = 1
            # negative log-likelihood loss
            loss = -log_probs[index]
            log_prob_sequence.append(log_probs[index])
            loss.backward()
            optimizer.step()
        final.append(probabilities[-1][1].item())
        hist, _ = np.histogram(final, range=(0,1), bins=20)
        print(text_barchart(hist, high=np.amax(hist)), run)
        log_probs = torch.tensor(log_prob_sequence)
        # for i, p in enumerate(torch.exp(log_probs).tolist()):
        #     if i % 10 == 0:
        #         end = '\n'
        #     else:
        #         end = ' '
        #     print(f"{p:2.4f}", end=end)
        # print()
        before = torch.sum(log_probs)
        print(before)
        log_prob_sequence: list[float] = []
        probabilities = []
        np.random.shuffle(flips)
        for step, flip in enumerate(flips):
            optimizer_2.zero_grad()
            log_probs = model_2()
            probs = torch.exp(log_probs)
            probabilities.append(probs)
            if debug and step % (steps // 100) == 0:
                print(f"{probs[0].item(): 3.3f} {probs[1].item(): 3.3f}")
            if flip:
                index = 0
            else:
                index = 1
            # negative log-likelihood loss
            loss = -log_probs[index]
            log_prob_sequence.append(log_probs[index])
            loss.backward()
            optimizer_2.step()
        log_probs = torch.tensor(log_prob_sequence)
        after = torch.sum(log_probs)
        print(after)
        print(after / before)
    # print(probabilities)
