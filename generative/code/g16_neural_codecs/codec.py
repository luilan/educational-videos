"""A tiny neural audio codec in the style of SoundStream and EnCodec, trained on a CPU.

    python3 codec.py              # ~2 hours on 16 CPU cores → out/codec.pt, out/train_log.json

encoder:   24,000 samples/s → strided convolutions (2·4·5·8 = 320) → 75 vectors of 64 numbers per second
quantizer: residual vector quantization, 8 codebooks of 1,024 vectors → 8 integers (tokens) per frame
decoder:   the mirror image, transposed convolutions back to 24,000 samples/s
loss:      multi-scale mel-spectrogram distance + a small waveform term + the commitment term (no GAN)
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000
STRIDES = (2, 4, 5, 8)          # product 320 → 75 frames per second
HOP = math.prod(STRIDES)
DIM, N_Q, CODES = 64, 8, 1024   # 8 codebooks × 10 bits × 75 frames/s = 6 kbit/s
STEPS = int(os.environ.get("STEPS", 12000))
BATCH, SEG = 16, 40 * HOP       # 16 crops of 40 frames (0.53 s) per step


class ResUnit(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.c2(F.elu(self.c1(F.elu(x))))


class Encoder(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        layers = [nn.Conv1d(1, ch, 7, padding=3)]
        for s in STRIDES:
            layers += [ResUnit(ch, 1), ResUnit(ch, 3), nn.ELU(), nn.Conv1d(ch, 2 * ch, 2 * s, stride=s, padding=(s + 1) // 2)]
            ch *= 2
        layers += [nn.ELU(), nn.Conv1d(ch, DIM, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # (B, samples) → (B, DIM, frames)
        return self.net(x[:, None])


class Decoder(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        layers = [nn.Conv1d(DIM, ch, 7, padding=3)]
        for s in reversed(STRIDES):
            layers += [nn.ELU(), nn.ConvTranspose1d(ch, ch // 2, 2 * s, stride=s, padding=s // 2 + s % 2, output_padding=s % 2),
                       ResUnit(ch // 2, 1), ResUnit(ch // 2, 3)]
            ch //= 2
        layers += [nn.ELU(), nn.Conv1d(ch, 1, 7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):                       # (B, DIM, frames) → (B, samples)
        return self.net(z)[:, 0]


class Codebook(nn.Module):
    """One codebook. Vectors move toward the encoder outputs assigned to them (moving averages, decay 0.99)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("embed", torch.zeros(CODES, DIM))
        self.register_buffer("count", torch.ones(CODES))
        self.register_buffer("sums", torch.zeros(CODES, DIM))
        self.register_buffer("ready", torch.tensor(False))

    def nearest(self, x):                       # x: (N, DIM) → index of the closest codebook vector
        d = (x * x).sum(1, keepdim=True) - 2 * x @ self.embed.T + (self.embed * self.embed).sum(1)
        return d.argmin(1)

    def forward(self, x):
        if self.training and not self.ready:    # start from real encoder outputs
            self.embed.copy_(x[torch.randint(len(x), (CODES,))].detach() + 1e-3 * torch.randn(CODES, DIM))
            self.sums.copy_(self.embed)
            self.ready.fill_(True)
        idx = self.nearest(x)
        q = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, CODES).float()
                self.count.mul_(0.99).add_(onehot.sum(0), alpha=0.01)
                self.sums.mul_(0.99).add_(onehot.T @ x, alpha=0.01)
                n = self.count.sum()
                smoothed = (self.count + 1e-5) / (n + CODES * 1e-5) * n
                self.embed.copy_(self.sums / smoothed[:, None])
                dead = self.count < 0.05           # unused vectors restart at random encoder outputs
                if dead.any():
                    k = int(dead.sum())
                    pick = x[torch.randint(len(x), (k,))]
                    self.embed[dead] = pick
                    self.sums[dead] = pick
                    self.count[dead] = 1.0
        return q, idx


class RVQ(nn.Module):
    """Residual vector quantization: each codebook quantizes what the previous ones missed."""

    def __init__(self):
        super().__init__()
        self.books = nn.ModuleList(Codebook() for _ in range(N_Q))

    def forward(self, z, n_q=N_Q):              # z: (B, DIM, frames)
        B, D, T = z.shape
        x = z.permute(0, 2, 1).reshape(-1, D)
        residual, total, codes, commit = x, torch.zeros_like(x), [], 0.0
        for book in self.books[:n_q]:
            q, idx = book(residual.detach() if self.training else residual)
            commit = commit + F.mse_loss(residual, q.detach())
            total = total + q
            residual = residual - q.detach()
            codes.append(idx.view(B, T))
        q = x + (total - x).detach()            # straight-through: gradients skip the rounding
        return q.view(B, T, D).permute(0, 2, 1), torch.stack(codes, 1), commit

    def decode_codes(self, codes):              # (B, n_q, frames) → (B, DIM, frames)
        return sum(self.books[i].embed[codes[:, i]] for i in range(codes.shape[1])).permute(0, 2, 1)


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.rvq, self.decoder = Encoder(), RVQ(), Decoder()

    def forward(self, x, n_q=N_Q):
        z = self.encoder(x)
        q, codes, commit = self.rvq(z, n_q)
        return self.decoder(q)[:, :x.shape[1]], codes, commit

    @torch.no_grad()
    def encode(self, x, n_q=N_Q):               # samples → integer tokens (n_q, frames)
        pad = (-len(x)) % HOP
        z = self.encoder(F.pad(torch.as_tensor(x, dtype=torch.float32), (0, pad))[None])
        return self.rvq(z, n_q)[1][0]

    @torch.no_grad()
    def decode(self, codes, length=None):       # tokens → samples
        y = self.decoder(self.rvq.decode_codes(codes[None]))[0]
        return y[:length] if length else y


def mel_matrix(n_fft, n_mels, sr=SR):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(0, mel(sr / 2), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    M = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, c, hi = edges[i:i + 3]
        M[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return torch.from_numpy(M)


SCALES = [(512, 40), (1024, 80), (2048, 80)]
MELS = {n: mel_matrix(n, m) for n, m in SCALES}


def log_mel(x, n_fft):
    S = torch.stft(x, n_fft, n_fft // 4, window=torch.hann_window(n_fft), return_complex=True).abs() ** 2
    return torch.log(MELS[n_fft] @ S + 1e-5)


def mel_loss(y, x):
    return sum(F.l1_loss(log_mel(y, n), log_mel(x, n)) for n, _ in SCALES) / len(SCALES)


def crops(audio, rng):
    starts = rng.integers(0, len(audio) - SEG, BATCH)
    return torch.from_numpy(np.stack([audio[s:s + SEG] for s in starts]))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", 14)))
    rng = np.random.default_rng(0)
    audio = np.load(os.path.join(HERE, "data/train.npy")).astype(np.float32)
    model = Codec()
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"encoder {n_enc:,} parameters, decoder {n_dec:,}; {len(audio) / SR / 60:.1f} min of training audio")
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()), 6e-4, betas=(0.8, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    log, t0 = [], time.time()
    for step in range(STEPS + 1):
        x = crops(audio, rng)
        n_q = int(rng.integers(1, N_Q + 1))     # quantizer dropout: one model for every bitrate
        y, codes, commit = model(x, n_q)
        lm = mel_loss(y, x)
        loss = lm + 0.1 * F.l1_loss(y, x) + 0.25 * commit
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            used = [int(torch.unique(codes[:, i]).numel()) for i in range(codes.shape[1])]
            log.append({"step": step, "n_q": n_q, "mel": round(lm.item(), 4), "commit": round(float(commit), 4),
                        "codes_used_first_book": used[0], "minutes": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
        if step % 1000 == 0 or step == STEPS:
            torch.save(model.state_dict(), os.path.join(HERE, "out/codec.pt"))
            json.dump({"encoder_parameters": n_enc, "decoder_parameters": n_dec, "steps": STEPS, "batch": BATCH,
                       "segment_s": SEG / SR, "train_minutes": len(audio) / SR / 60, "log": log},
                      open(os.path.join(HERE, "out/train_log.json"), "w"), indent=1)
