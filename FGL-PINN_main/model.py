import torch
import torch.nn as nn
import math

class FourierEncoder(nn.Module):
    def __init__(self, num_fourier_frequencies=5, scale=1.0):
        super().__init__()
        self.num_fourier_frequencies = num_fourier_frequencies
        self.scale = scale
        self.register_buffer("frequencies", 2.0 ** torch.linspace(0.0, num_fourier_frequencies - 1, num_fourier_frequencies))

    def forward(self, x):
        # x: [B, 1]
        x = x * self.scale
        freq_terms = x * self.frequencies  # [B, F]
        return torch.cat([torch.sin(freq_terms), torch.cos(freq_terms)], dim=-1)  # [B, 2F]

class FNN(nn.Module):
    def __init__(self, layers, activation, num_fourier_frequencies=5, scale=1.0, in_tf=None, out_tf=None):
        super().__init__()
        self.activation = activation
        self.in_tf = in_tf
        self.out_tf = out_tf
        self.encoder = FourierEncoder(num_fourier_frequencies, scale)

        input_dim = 4 * 2 * num_fourier_frequencies
        adjusted_layers = [input_dim] + layers

        print(f"FNN input dimension: {input_dim}, adjusted_layers: {adjusted_layers}")

        self.linears = nn.ModuleList()
        for i in range(len(adjusted_layers) - 1):
            linear = nn.Linear(adjusted_layers[i], adjusted_layers[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)

    def forward(self, inputs):
        # inputs: [B, 4]
        if self.in_tf:
            inputs = self.in_tf(inputs)

        # Apply Fourier encoding to each coordinate
        encoded_list = [self.encoder(inputs[:, i:i+1]) for i in range(inputs.shape[1])]
        X = torch.cat(encoded_list, dim=-1)  # [B, 4×2F]

        if not hasattr(self, '_logged_shape'):
            print(f"Fourier encoded shape: {X.shape}")
            self._logged_shape = True

        for linear in self.linears[:-1]:
            X = self.activation(linear(X))
        X = self.linears[-1](X)

        if self.out_tf:
            X = self.out_tf(X)
        return X
