import torch
import math
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


def main():
    model = Network()
    model_2 = Network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_2 = torch.optim.AdamW(model_2.parameters())
    model_state = model.state_dict()
    model_2.load_state_dict(model_state)
    sampled = []
    log_prob_sequence: list[float] = []
    probabilities = []
    steps = 10000
    for step in range(steps):
        optimizer.zero_grad()
        log_probs = model()
        probs = torch.exp(log_probs)
        r = random()
        probabilities.append(probs[1].item())
        if step % (steps // 100) == 0:
            print(probs[1].item())
        if r <= probs[0]:
            loss = probs[1]
            log_prob_sequence.append(log_probs[0])
            sampled.append(0)
        else:
            loss = probs[0]
            log_prob_sequence.append(log_probs[1])
            sampled.append(1)
        loss.backward()
        optimizer.step()
    log_probs = torch.tensor(log_prob_sequence)
    print(torch.sum(log_probs))
    # print(probabilities)
