import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

def _per_sample_pca_rgb_maps(features: torch.Tensor) -> torch.Tensor:
    """Project normalized feature maps to RGB with PCA independently per sample.

    Running PCA per image avoids batch-composition effects where changing batch size
    can alter the fitted PCA basis and therefore change colors for the same sample.
    """
    batch_size, channels, height, width = features.shape
    normalized = F.normalize(features, p=2, dim=1, eps=1e-6).permute(0, 2, 3, 1)
    normalized_np = normalized.detach().cpu().numpy()

    rgb_maps = torch.empty((batch_size, height, width, 3), dtype=torch.float32)

    for sample_idx in range(batch_size):
        sample_flat = normalized_np[sample_idx].reshape(-1, channels)  # [H*W, C]

        # Guard tiny spatial maps where H*W < 3 by projecting to available components
        # and zero-padding missing channels.
        n_components = max(1, min(3, sample_flat.shape[0], sample_flat.shape[1]))
        pca = PCA(n_components=n_components, whiten=True)
        sample_pca = pca.fit_transform(sample_flat)

        if n_components < 3:
            padded = torch.zeros((sample_pca.shape[0], 3), dtype=torch.float32)
            padded[:, :n_components] = torch.from_numpy(sample_pca).float()
            sample_pca_tensor = padded
        else:
            sample_pca_tensor = torch.from_numpy(sample_pca).float()

        rgb_maps[sample_idx] = sample_pca_tensor.view(height, width, 3)

    return torch.sigmoid(rgb_maps.mul(2.0))


def rgb_pca_maps_for_features(student_features: torch.Tensor, teacher_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate RGB maps of PCA for student and teacher features for visualization.
    
    Args:
        student_features: tensor [B, C, H, W] of student features
        teacher_features: tensor [B, C, H, W] of teacher features
    Returns:
        Tuple of (student_rgb_maps, teacher_rgb_maps) where each is a tensor [B, H, W, 3] of RGB maps
    """
    student_rgb_maps = _per_sample_pca_rgb_maps(student_features)
    teacher_rgb_maps = _per_sample_pca_rgb_maps(teacher_features)
    return student_rgb_maps, teacher_rgb_maps