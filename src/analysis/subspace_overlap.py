"""
Subspace overlap analysis for Phase 5.
Computes principal angles between ROME and LoRA weight-update subspaces.
"""

import numpy as np
import torch


def principal_angles(delta_W_a: np.ndarray, delta_W_b: np.ndarray) -> np.ndarray:
    """
    Computes principal angles (radians) between the column spaces of two
    weight-update matrices of the same shape.
    
    Returns array of angles, sorted ascending (0 = perfectly aligned direction exists).
    
    Args:
        delta_W_a: First weight update matrix (out_features, in_features)
        delta_W_b: Second weight update matrix (out_features, in_features)
    
    Returns:
        Array of principal angles in radians, sorted ascending
    """
    # SVD of both matrices
    Ua, Sa, _ = np.linalg.svd(delta_W_a, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(delta_W_b, full_matrices=False)
    
    # Effective rank: number of singular values above threshold
    rank_a = np.sum(Sa > 1e-10)
    rank_b = np.sum(Sb > 1e-10)
    
    if rank_a == 0 or rank_b == 0:
        return np.array([np.pi / 2])  # orthogonal if one is zero
    
    # Keep only significant singular vectors
    Ua = Ua[:, :rank_a]
    Ub = Ub[:, :rank_b]
    
    # Principal angles via SVD of Ua^T @ Ub
    _, s, _ = np.linalg.svd(Ua.T @ Ub)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    
    return np.sort(angles)


def subspace_overlap_score(delta_W_a: np.ndarray, delta_W_b: np.ndarray) -> float:
    """
    Subspace overlap score based on smallest principal angle.
    1.0 = identical subspace, 0.0 = orthogonal.
    
    Args:
        delta_W_a: First weight update matrix
        delta_W_b: Second weight update matrix
    
    Returns:
        Cosine of the smallest principal angle (overlap score in [0, 1])
    """
    angles = principal_angles(delta_W_a, delta_W_b)
    return float(np.cos(angles[0]))


def subspace_overlap_score_all_angles(delta_W_a: np.ndarray, delta_W_b: np.ndarray) -> dict:
    """
    Returns detailed principal angle analysis.
    
    Returns:
        Dict with angles (radians), cosines, and overlap score
    """
    angles = principal_angles(delta_W_a, delta_W_b)
    cosines = np.cos(angles)
    return {
        'angles_rad': angles.tolist(),
        'angles_deg': (angles * 180 / np.pi).tolist(),
        'cosines': cosines.tolist(),
        'overlap_score': float(cosines[0]),
        'mean_cosine': float(np.mean(cosines)),
        'min_angle_deg': float(angles[0] * 180 / np.pi),
        'max_angle_deg': float(angles[-1] * 180 / np.pi),
    }


def frobenius_norm(tensor: torch.Tensor) -> float:
    """Computes Frobenius norm of a tensor."""
    return tensor.norm(p='fro').item()


def compute_rome_delta_W(model, target_class: int, layer_name: str = 'fc2') -> torch.Tensor:
    """
    Recompute the ROME delta_W for a given model, target class, and layer.
    Matches the ROME computation in multiclass_rome.py.
    
    Args:
        model: The trained model
        target_class: The target class for ROME edit
        layer_name: 'fc1' or 'fc2'
    
    Returns:
        The ROME delta weight update matrix
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from multiclass_rome import compute_rome_edit
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    device = next(model.parameters()).device
    
    # Get test dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    delta, u, v, used_layer = compute_rome_edit(model, test_loader, device, target_class, layer_name)
    if delta is None:
        return torch.zeros_like(getattr(model, used_layer).weight)
    return delta


def compute_lora_delta_W(lora_layer) -> torch.Tensor:
    """
    Extract the learned LoRA delta_W from a LoRALinear layer.
    
    Args:
        lora_layer: A LoRALinear module
    
    Returns:
        The delta weight matrix (out_features, in_features)
    """
    return lora_layer.delta_W()