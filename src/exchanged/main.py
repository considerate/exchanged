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
    #model_2 = Network()
    #optimizer_2 = torch.optim.AdamW(model_2.parameters())
    #model_state = model.state_dict()
    #model_2.load_state_dict(model_state)
    steps = 10000
    runs = 1000
    final = []
    for run in range(runs):
        model = Network()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        sampled = []
        log_prob_sequence: list[float] = []
        probabilities = []
        for step in range(steps):
            optimizer.zero_grad()
            log_probs = model()
            probs = torch.exp(log_probs)
            probabilities.append(probs)
            r = random()
            if step % (steps // 100) == 0:
                print(f"{probs[0].item(): 3.3f} {probs[1].item(): 3.3f}")
            if r <= probs[0]:
                index = 0
            else:
                index = 1
            # negative log-likelihood loss
            loss = -log_probs[index]
            log_prob_sequence.append(log_probs[index])
            sampled.append(index)
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
        print(torch.sum(log_probs))
    # print(probabilities)
