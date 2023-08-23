import numpy as np

class FedAvg:
    def __init__(self):
        self.weights = []
        self.num_samples = []

    def add_client(self, client_weights, num_samples):
        self.weights.append(client_weights)
        self.num_samples.append(num_samples)

    def fedavg(self):
        # FedAvg with weight
        total_samples = sum(self.num_samples)
        base = [0] * len(self.weights[0])
        for i, client_weight in enumerate(self.weights):
            total_samples += self.num_samples[i]
            for j, v in enumerate(client_weight):
                base[j] += (self.num_samples[i] / total_samples * v.astype(np.float64))
        # Update the model
        return base


