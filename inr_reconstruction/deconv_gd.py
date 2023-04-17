import torch

class GradientDescentDeconvolution(torch.nn.Module):
    def __init__(self, num_batch, num_bins, dev):
        super().__init__()
        self.time_bins = 1e-9*torch.randn((num_batch, num_bins)).to(dev)

        self.time_bins.requires_grad_(True)

    def forward(self):
        return self.time_bins

    def get_params(self, lr):
        return [{'params': self.time_bins, 'lr': lr}]
