"""
LoRA (Low-Rank Adaptation) wrapper for frozen linear layers.
Used in Phase 5 for gradient-learned low-rank correction comparison with ROME.
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a trainable low-rank correction:
    output = base(x) + scaling * (x @ A^T @ B^T)
    
    Zero-initializing B guarantees Delta_W = 0 at initialization,
    making the wrapped model numerically identical to the base checkpoint.
    """
    
    def __init__(
        self, 
        base_linear: nn.Linear, 
        rank: int, 
        alpha: Optional[float] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = (alpha / rank) if alpha is not None else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # A: (rank, in_features) - small random init
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # B: (out_features, rank) - zero init -> delta_W = 0 initially
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.dropout is not None:
            x = self.dropout(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * delta

    def delta_W(self) -> torch.Tensor:
        """Returns the effective weight update, same shape as base.weight."""
        with torch.no_grad():
            return self.scaling * (self.lora_B @ self.lora_A)

    def get_lora_params(self) -> list:
        """Returns LoRA parameters for optimizer."""
        return [self.lora_A, self.lora_B]


def attach_lora(model: nn.Module, layer_name: str, rank: int, alpha: Optional[float] = None) -> LoRALinear:
    """
    Attaches LoRA to a named linear layer in the model.
    Replaces the attribute in-place and returns the LoRA module.
    
    Args:
        model: The model containing the layer
        layer_name: Name of the layer attribute ('fc1', 'fc2', 'fc3')
        rank: LoRA rank
        alpha: LoRA scaling factor (alpha/rank)
    
    Returns:
        The attached LoRALinear module
    """
    base = getattr(model, layer_name)
    lora_layer = LoRALinear(base, rank=rank, alpha=alpha)
    setattr(model, layer_name, lora_layer)
    return lora_layer


def detach_lora(model: nn.Module, layer_name: str) -> nn.Linear:
    """
    Removes LoRA wrapper and restores the original base linear layer.
    
    Args:
        model: The model containing the LoRA-wrapped layer
        layer_name: Name of the layer attribute
    
    Returns:
        The original base linear layer
    """
    lora_layer = getattr(model, layer_name)
    if isinstance(lora_layer, LoRALinear):
        base = lora_layer.base
        setattr(model, layer_name, base)
        return base
    return lora_layer