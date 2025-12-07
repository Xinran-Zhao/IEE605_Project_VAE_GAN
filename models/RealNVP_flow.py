import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ActNorm(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def initialize(self, x: torch.Tensor):
        with torch.no_grad():
            # x: (B,C,H,W)
            mean = x.mean(dim=[0,2,3], keepdim=True)  # (1,C,1,1)
            std = x.std(dim=[0,2,3], keepdim=True)
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(torch.log(1.0 / (std + 1e-6)))

    def forward(self, x: torch.Tensor, reverse: bool = False):
        # returns (x_out, logdet_per_example)
        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        _, _, H, W = x.shape
        if reverse:
            x = (x - self.bias) * torch.exp(-self.log_scale)
            ld = -self.log_scale.sum() * H * W
        else:
            x = x * torch.exp(self.log_scale) + self.bias
            ld = self.log_scale.sum() * H * W
        # ld is scalar (sum over channels); we'll return per-example logdet as vector
        # convert to tensor of shape (B,)
        ld = ld * torch.ones(x.shape[0], device=x.device)
        return x, ld

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        w_init = np.linalg.qr(np.random.randn(num_channels, num_channels))[0].astype(np.float32)
        W = torch.from_numpy(w_init)
        self.W = nn.Parameter(W)  # (C, C)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        B, C, H, W = x.shape
        if reverse:
            W_inv = torch.inverse(self.W).to(x.device)
            x = F.conv2d(x, W_inv.view(C, C, 1, 1))
            ld = -H * W * torch.slogdet(self.W)[1]
        else:
            x = F.conv2d(x, self.W.view(C, C, 1, 1))
            ld = H * W * torch.slogdet(self.W)[1]
        ld = ld * torch.ones(B, device=x.device)
        return x, ld


def checkerboard_mask(h: int, w: int, parity: int):
    grid = np.indices((h,w)).sum(axis=0) % 2
    if parity:
        grid = 1 - grid
    mask = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return mask  # caller should move to correct device and expand channels

class ConvSubnet(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 64, out_channels: int = None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )
        # init last conv to zeros so initial s ~ 0, t ~ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor):
        return self.net(x)

class AffineCoupling(nn.Module):
    def __init__(self, num_channels: int, hidden_channels: int = 64, use_checkerboard: bool = True, parity: int = 0):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.use_checkerboard = use_checkerboard
        self.parity = parity
        
        if use_checkerboard:
            self.st_net = ConvSubnet(in_channels=num_channels, mid_channels=hidden_channels, out_channels=num_channels*2)
        else:
            c_half = num_channels // 2
            self.st_net = ConvSubnet(in_channels=c_half, mid_channels=hidden_channels, out_channels=c_half*2)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        B, C, H, W = x.shape

        if self.use_checkerboard:
            mask = checkerboard_mask(H, W, self.parity).to(x.device)  # (1,1,H,W)
            mask = mask.expand(B, C, H, W)  # (B,C,H,W)
            x_masked = x * mask
            st = self.st_net(x_masked)
            s, t = st.chunk(2, dim=1)
            # limit scale to stable range
            s = torch.tanh(s) * 2.0
            if reverse:
                # x = mask*x + ( (x - t) * exp(-s) ) * (1-mask)
                x = x * mask + ( (x - t) * torch.exp(-s) ) * (1.0 - mask)
                ld = -(s * (1.0 - mask)).sum(dim=[1,2,3])
            else:
                x = x * mask + ( x * torch.exp(s) + t ) * (1.0 - mask)
                ld = (s * (1.0 - mask)).sum(dim=[1,2,3])
            return x, ld
        else:
            # split channels
            c_half = C // 2
            x_a = x[:, :c_half, :, :]
            x_b = x[:, c_half:, :, :]
            st = self.st_net(x_a)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s) * 2.0
            if reverse:
                x_b = (x_b - t) * torch.exp(-s)
                ld = -s.view(B, -1).sum(dim=1)
            else:
                x_b = x_b * torch.exp(s) + t
                ld = s.view(B, -1).sum(dim=1)
            x = torch.cat([x_a, x_b], dim=1)
            return x, ld


class FlowStep(nn.Module):
    def __init__(self, num_channels: int, hidden_channels: int = 64, use_checkerboard: bool = True, parity: int = 0):
        super().__init__()
        self.actnorm = ActNorm(num_channels)
        self.invconv = Invertible1x1Conv(num_channels)
        self.coupling = AffineCoupling(num_channels, hidden_channels=hidden_channels, use_checkerboard=use_checkerboard, parity=parity)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        ld_total = x.new_zeros(x.shape[0])
        if reverse:
            x, ld = self.coupling(x, reverse=True); ld_total = ld_total + ld
            x, ld = self.invconv(x, reverse=True); ld_total = ld_total + ld
            x, ld = self.actnorm(x, reverse=True); ld_total = ld_total + ld
        else:
            x, ld = self.actnorm(x, reverse=False); ld_total = ld_total + ld
            x, ld = self.invconv(x, reverse=False); ld_total = ld_total + ld
            x, ld = self.coupling(x, reverse=False); ld_total = ld_total + ld
        return x, ld_total

class RealNVP(nn.Module):
    def __init__(self, in_channels: int = 3, num_flows: int = 8, hidden_channels: int = 64, use_checkerboard: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_flows = num_flows
        self.hidden_channels = hidden_channels
        self.use_checkerboard = use_checkerboard

        layers = []
        for i in range(num_flows):
            parity = i % 2
            layers.append(FlowStep(in_channels, hidden_channels=hidden_channels, use_checkerboard=use_checkerboard, parity=parity))
        self.flow = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        ld_total = x.new_zeros(batch)
        z = x
        for step in self.flow:
            z, ld = step(z, reverse=False)
            ld_total = ld_total + ld
        return z, ld_total

    def inverse(self, z: torch.Tensor):
        x = z
        ld_total = z.new_zeros(z.shape[0])
        for step in reversed(self.flow):
            x, ld = step(x, reverse=True)
            ld_total = ld_total + ld
        return x, ld_total

    def log_prob(self, x: torch.Tensor):
        z, ld = self.forward(x)
        # Gaussian log-likelihood per element
        prior_ll = -0.5 * (z ** 2 + math.log(2 * math.pi))
        prior_ll = prior_ll.sum(dim=[1,2,3])  # sum over C,H,W
        return prior_ll + ld

    def sample(self, num_samples: int, H: int, W: int, device: torch.device = None):
        device = device or torch.device("cpu")
        z = torch.randn(num_samples, self.in_channels, H, W, device=device)
        x, _ = self.inverse(z)
        return x

