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

class Minimal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.tensor(0.0),
            requires_grad=True,
        )
    def forward(self):
        p = torch.nn.functional.sigmoid(self.param).unsqueeze(0)
        probs = torch.concatenate([1-p, p])
        return torch.log(probs)

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

def main_polya():
    runs = [int(10**((i+10)/10)) for i in range(31)]
    bias = 0.1
    debug = False
    totals = []
    lens = []
    mean_log_proabilities = []
    mean_head_probs = []
    for run, steps in enumerate(runs):
        print(f"\n\n\nNEW RUN OF {steps} STEPS!!!\n\n\n")
        lens.append(steps)
        final = []
        ratios = []
        log_probabilities = []
        head_probs = []
        for _ in range(10):
            lr=1e-2
            white, black = 1, 1
            log_prob_sequence: list[float] = []
            flips = [ random () <= bias for _ in range(steps)]
            prob = 0
            for step, flip in enumerate(flips):
                if flip:
                    white += 1
                    index = 0
                else:
                    black += 1
                    index = 1
                log_prob = math.log(white) - math.log(white + black)
                log_prob_2 = math.log(black) - math.log(white + black)
                prob = math.exp(log_prob)
                log_probs = torch.tensor([log_prob, log_prob_2])
                # negative log-likelihood loss
                log_prob_sequence.append(log_probs[index].item())
            head_probs.append(prob)
            final.append(prob)
            hist, _ = np.histogram(final, range=(0,1), bins=100)
            print(text_barchart(hist, high=np.amax(hist)), run)
            log_probs = torch.tensor(log_prob_sequence)
            before = torch.sum(log_probs)
            print(before, prob)
            log_prob_sequence: list[float] = []
            np.random.shuffle(flips)
            white, black = 1, 1
            prob = 0
            for step, flip in enumerate(flips):
                if flip:
                    white += 1
                    index = 0
                else:
                    black += 1
                    index = 1
                prob = white / (white + black)
                log_probs = torch.log(torch.tensor([prob, 1-prob]))
                # negative log-likelihood loss
                log_prob_sequence.append(log_probs[index].item())
            log_probs = torch.tensor(log_prob_sequence)
            after = torch.sum(log_probs)
            head_probs.append(prob)
            final.append(prob)
            print(after, prob)
            ratio = before - after
            ratios.append(ratio)
            log_probabilities.append(before)
            log_probabilities.append(after)
            print(f"{ratio.item():6.4g}")
            print(f"{np.exp(ratio.item()):6.4g}")
        ratios_array = np.array(ratios)
        rms = np.sqrt(np.sum((ratios_array ** 2)) / len(ratios))
        totals.append(rms)
        mean_log_proabilities.append(np.mean(log_probabilities))
        mean_head_probs.append(np.mean(head_probs))
        for c, t, mean_log, mean_head in zip(lens, totals, mean_log_proabilities, mean_head_probs, strict=True):
            print(
                c,
                f"{t:8.4f}",
                f"{t/c:8.4g}",
                f"{mean_log:8.4g}",
                f"{mean_log/c:8.4g}",
                f"{np.exp(mean_log/c):8.4g}",
                f"{mean_head:4.3f}",
            )

def main_network():
    runs = [int(10**((i+10)/10)) for i in range(31)]
    bias = 0.1
    debug = False
    totals = []
    lens = []
    mean_log_proabilities = []
    mean_head_probs = []
    for run, steps in enumerate(runs):
        print(f"\n\n\nNEW RUN OF {steps} STEPS!!!\n\n\n")
        lens.append(steps)
        final = []
        ratios = []
        log_probabilities = []
        head_probs = []
        for _ in range(10):
            lr=1e-2
            model = Network()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            model_2 = Network()
            optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=lr)
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
            prob = probabilities[-1][0].item()
            head_probs.append(prob)
            final.append(prob)
            hist, _ = np.histogram(final, range=(0,1), bins=100)
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
            print(before, prob)
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
            prob = probabilities[-1][0].item()
            head_probs.append(prob)
            final.append(prob)
            print(after, prob)
            ratio = before - after
            ratios.append(ratio)
            log_probabilities.append(before)
            log_probabilities.append(after)
            print(f"{ratio.item():6.4g}")
            print(f"{np.exp(ratio.item()):6.4g}")
        ratios_array = np.array(ratios)
        rms = np.sqrt(np.sum((ratios_array ** 2)) / len(ratios))
        totals.append(rms)
        mean_log_proabilities.append(np.mean(log_probabilities))
        mean_head_probs.append(np.mean(head_probs))
        for c, t, mean_log, mean_head in zip(lens, totals, mean_log_proabilities, mean_head_probs, strict=True):
            print(
                c,
                f"{t:8.4f}",
                f"{t/c:8.4g}",
                f"{mean_log/c:8.4g}",
                f"{np.exp(mean_log/c):8.4g}",
                f"{mean_head:4.3f}",
            )
    # print(probabilities)

def main():
    #main_network()
    main_polya()
