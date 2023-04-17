import tinycudann as tcnn
import torch
from sas_utils import safe_normalize
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils import custom_grid_sample_3d
import constants as c

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            # Keep bias false
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Network(torch.nn.Module):
    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def no_network_model(self, x):
        # Grid sample doesn't support complex grid with floating point input so have to split
        # into real and imag
        #x[..., 1] = -x[..., 1]
        #x[..., 2] = -x[..., 2]

        self.model_real = partial(custom_grid_sample_3d,
                                 self.scene_voxels.real)
        self.model_imag = partial(custom_grid_sample_3d,
                                 self.scene_voxels.imag)

        return torch.complex(real=self.model_real(x), imag=self.model_imag(x))

    def __init__(self, inr_config, dev, num_layers, num_neurons, incoherent=False, real_only=False,
                 scene_voxels=None):
        super().__init__()
        self.dev = dev
        self.no_network = False
        self.scene_voxels = scene_voxels


        if self.scene_voxels is not None:
            assert self.scene_voxels.ndim == 5
            self.scene_voxels.requires_grad_(True)
            self.no_network = True

            self.model = self.no_network_model

        else:
            self.incoherent = incoherent
            self.real_only = real_only

            if self.incoherent or self.real_only:
                self.n_output_dims = 1
            else:
                self.n_output_dims = 2

            self.n_input_dims = 3

            self.encoder = tcnn.Encoding(self.n_input_dims, inr_config['encoding'])
            self.network = MLP(dim_in=self.encoder.n_output_dims,
                               dim_out=self.n_output_dims,
                               dim_hidden=num_neurons,
                               num_layers=num_layers).float()

            self.model = torch.nn.Sequential(self.encoder, self.network).to(self.dev)

    def model_out(self, x):

        if self.no_network:
            x_input = x[None, None, None, ...]
            out = self.model(x_input)
            out = out.squeeze().view(-1)
            return out
        else:
            out = self.encoder(x)
            out = self.network(out.float())

            if self.incoherent or self.real_only:
                #out = torch.nn.functional.relu(out).float()
                out = out.float()
            else:
                out = torch.complex(real=out[..., 0].float(), imag=out[..., 1].float())

            return out

    def forward(self, coords_to, compute_normals=True):
        with torch.enable_grad():
            # Track gradient for all coordinates

            coords_to.requires_grad_(True)

            coords_to_network = coords_to.reshape(-1, 3)

            output = self.model_out(coords_to_network)

            complex_scatterer_to = output

            if compute_normals:
                normals_out = - torch.autograd.grad(torch.sum(output.abs()),
                                                coords_to_network, create_graph=True)[0]
                normals_out = safe_normalize(normals_out).float()
                normals_out[torch.isnan(normals_out)] = 0

                normals = normals_out
            else:
                normals = None

        return {
            'scatterers_to': complex_scatterer_to,
            'normals': normals
        }

    def normal(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            comp_albedo = self.model_out(x)

            normal = - torch.autograd.grad(torch.sum(comp_albedo.abs()), x, create_graph=True)[0]
            normal = safe_normalize(normal).float()
            normal[torch.isnan(normal)] = 0

        return comp_albedo, normal

    def get_params(self, lr):
        if self.no_network:
            return [{'params': self.scene_voxels, 'lr': lr}]
        else:
            return [{'params': self.model.parameters(), 'lr': lr}]
