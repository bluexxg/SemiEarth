"""Training-only, per-image channel cross-attention (not spatial N x N attention)."""
import torch
from torch import nn
from torch.nn import functional as F


class CTSA(nn.Module):
    def __init__(self, channels, dim=128, heads=4, alpha=0.1,
                 temperature=0.1, gate_statistics=False):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or temperature <= 0:
            raise ValueError('CTSA requires positive dim/heads/temperature and dim % heads == 0')
        self.heads, self.dim = heads, dim
        self.alpha, self.temperature = alpha, temperature
        self.gate_statistics = gate_statistics
        self.student_norm = nn.LayerNorm(channels)
        self.teacher_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, dim, bias=False)
        self.k = nn.Linear(channels, dim, bias=False)
        self.v = nn.Linear(channels, dim, bias=False)
        self.out = nn.Linear(dim, channels, bias=False)

    def forward(self, student, teacher, gate=None):
        if student.shape != teacher.shape or student.ndim != 3:
            raise ValueError('CTSA expects matching [B, N, C] student/teacher features')
        b, n, _ = student.shape
        if gate is None:
            gate = student.new_ones(b, n, 1)
        if gate.shape != (b, n, 1):
            raise ValueError('CTSA gate must have shape [B, N, 1]')
        gate = gate.detach().float().clamp(0, 1)
        s = self.student_norm(student)
        t = self.teacher_norm(teacher.detach())  # projection parameters still learn

        def split(x):
            return x.reshape(b, n, self.heads, self.dim // self.heads).permute(0, 2, 3, 1)

        q, k, v = split(self.q(s)), split(self.k(t)), split(self.v(t))
        # Spatial reductions and cosine logits in fp32, including under AMP.
        with torch.autocast(device_type=student.device.type, enabled=False):
            q, k = q.float(), k.float()
            if self.gate_statistics:
                weight = gate.transpose(1, 2).unsqueeze(1).sqrt()
                q, k = q * weight, k * weight
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            attention = ((q @ k.transpose(-2, -1)) / self.temperature).softmax(-1)
            mixed = attention @ v.float()
        mixed = mixed.permute(0, 3, 1, 2).reshape(b, n, self.dim).to(v.dtype)
        residual = self.out(mixed)
        return student + self.alpha * gate.to(residual.dtype) * residual
