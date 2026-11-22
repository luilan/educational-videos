"""A small diffusion model for Shapes: noise schedule, U-Net denoiser, training step and samplers.

Used by episodes 6–8 (and, with labels, 10–11). Pixels live in [−1, 1]. The network sees a noisy picture x_t and the
step t, and predicts the noise ε that was added (the "ε-prediction" of DDPM, Ho et al. 2020).

    model = UNet()
    loss = training_loss(model, x0)                    # one batch
    x = sample_ddpm(model, 16)                         # 1,000 steps
    x = sample_ddim(model, 16, steps=50)               # 50 steps, deterministic
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000


def cosine_alpha_bar(s=0.008):
    """ᾱ_t for t = 1..T (episode 5's cosine schedule)."""
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    beta = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - beta, 0).float(), beta.float()


ALPHA_BAR, BETA = cosine_alpha_bar()


def timestep_embedding(t, dim):
    """Sines and cosines of the step at many frequencies, like a transformer's position encoding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.norm2, self.conv2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]            # tell every channel which step this is
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """32×32 → 16×16 → 8×8 and back, with skip connections between matching sizes.

    use_time=False hides the step from the network; use_skips=False removes the skip connections (episode 8).
    num_classes > 0 adds a label embedding (episodes 10–11); label `num_classes` means "no label".
    """

    def __init__(self, ch=(64, 128, 128), t_dim=128, use_time=True, use_skips=True, num_classes=0):
        super().__init__()
        self.use_time, self.use_skips, self.t_dim = use_time, use_skips, t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.label = nn.Embedding(num_classes + 1, t_dim) if num_classes else None
        self.inp = nn.Conv2d(3, ch[0], 3, padding=1)
        self.down1, self.down2 = ResBlock(ch[0], ch[0], t_dim), ResBlock(ch[1], ch[1], t_dim)
        self.pool1, self.pool2 = nn.Conv2d(ch[0], ch[1], 3, 2, 1), nn.Conv2d(ch[1], ch[2], 3, 2, 1)
        self.mid1, self.mid2 = ResBlock(ch[2], ch[2], t_dim), ResBlock(ch[2], ch[2], t_dim)
        k = 2 if use_skips else 1
        self.upc2, self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1), ResBlock(ch[1] * k, ch[1], t_dim)
        self.upc1, self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1), ResBlock(ch[0] * k, ch[0], t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 3, 3, padding=1))

    def forward(self, x, t, y=None):
        temb = self.t_mlp(timestep_embedding(t if self.use_time else torch.zeros_like(t), self.t_dim))
        if self.label is not None:
            temb = temb + self.label(y)
        h1 = self.down1(self.inp(x), temb)                   # 64×32×32
        h2 = self.down2(self.pool1(h1), temb)                # 128×16×16
        h = self.mid2(self.mid1(self.pool2(h2), temb), temb)  # 128×8×8
        h = self.upc2(h)
        h = self.up2(torch.cat([h, h2], 1) if self.use_skips else h, temb)
        h = self.upc1(h)
        h = self.up1(torch.cat([h, h1], 1) if self.use_skips else h, temb)
        return self.out(h)


def noise(x0, t, eps):
    """Episode 5's jump: x_t = √ᾱ_t·x0 + √(1 − ᾱ_t)·ε, for steps t in 1..T."""
    a = ALPHA_BAR[t - 1].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


def training_loss(model, x0, y=None):
    """Pick a random step for every picture, noise it, ask the model for the noise."""
    t = torch.randint(1, T + 1, (len(x0),))
    eps = torch.randn_like(x0)
    return F.mse_loss(model(noise(x0, t, eps), t, y), eps)


@torch.no_grad()
def predict_eps(model, x, t, y=None, guidance=None, null=None):
    tt = torch.full((len(x),), t, dtype=torch.long)
    if guidance is None:
        return model(x, tt, y)
    both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, null)]))
    cond, uncond = both.chunk(2)
    return uncond + guidance * (cond - uncond)               # classifier-free guidance (episode 11)


@torch.no_grad()
def sample_ddpm(model, n, generator=None, y=None, guidance=None, null=None, keep=()):
    """Ancestral sampling: start from pure noise, take all T small steps back. keep: steps to record."""
    x = torch.randn(n, 3, 32, 32, generator=generator)
    saved = {}
    for t in range(T, 0, -1):
        eps = predict_eps(model, x, t, y, guidance, null)
        a, ab, b = 1 - BETA[t - 1], ALPHA_BAR[t - 1], BETA[t - 1]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        if t > 1:
            var = b * (1 - ALPHA_BAR[t - 2]) / (1 - ab)       # the "posterior" variance
            x = mean + var.sqrt() * torch.randn(x.shape, generator=generator)
        else:
            x = mean
        if t - 1 in keep:
            saved[t - 1] = x.clamp(-1, 1).clone()
    return (x.clamp(-1, 1), saved) if keep else x.clamp(-1, 1)


@torch.no_grad()
def sample_ddim(model, n, steps=50, generator=None, y=None, guidance=None, null=None, x=None):
    """DDIM (Song et al. 2020) with η = 0: deterministic, and it can skip steps."""
    if x is None:
        x = torch.randn(n, 3, 32, 32, generator=generator)
    ts = torch.linspace(T, 1, steps).round().long().tolist()
    for i, t in enumerate(ts):
        eps = predict_eps(model, x, t, y, guidance, null)
        ab = ALPHA_BAR[t - 1]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)          # the model's guess of the picture
        ab_next = ALPHA_BAR[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x.clamp(-1, 1)
