from typing import List, Optional
import torch
from torch import nn
import timm 
from .repvit_registry import REPVIT_MODELS

REPVIT_DINO_MODELS = {
    'repvit_m1_1_dino': {
        'backbone_name': 'repvit_m1_1.dist_300e_in1k',
        'adapter_dim': 1024,
    },
    'repvit_m1_vit_small_plus': {
        'backbone_name': 'repvit_m1.dist_in1k',
        'adapter_dim': 0,
    }
}



class RepVitDINO(nn.Module):
    """
    RepVit model distilled from DINOv3.
    """

    def __init__(
        self,
        backbone_name: str,
        adapter_dim: Optional[int] = None,
        use_adapter: bool = True,
        feature_indices: Optional[List[int]] = None,
    ):
        """
        Initialize the RepVitDINO model.
        
        Args:
            backbone_name: Name of the RepVit backbone to load
            adapter_dim: Dimension of the adapter layer (required when use_adapter=True)
            use_adapter: If True, apply adapter to the last selected feature
            feature_indices: Feature indices to return. Defaults to [-1] (last feature only)
        """
        super(RepVitDINO, self).__init__()
        self.use_adapter = use_adapter
        self.feature_indices = feature_indices if feature_indices is not None else [-1]
        
       
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
        )
        
        # 1x1 adapter conv
        repvit_cfg = next(
            (cfg for cfg in REPVIT_MODELS.values() if cfg.get('model_name') == backbone_name),
            None,
        )
        if repvit_cfg is None:
            raise ValueError(f"Backbone '{backbone_name}' not found in REPVIT_MODELS registry.")

        stage_channels = [stage['channels'] for stage in repvit_cfg['stages']]
        selected_channels = [stage_channels[idx] for idx in self.feature_indices]
        in_channels = selected_channels[-1]
        if self.use_adapter:
            if adapter_dim is None:
                raise ValueError("adapter_dim must be provided when use_adapter=True")
            self.adapter = nn.Conv2d(in_channels, adapter_dim, kernel_size=1)
        else:
            self.adapter = nn.Identity()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_path: str,  # required
        feature_indices: Optional[List[int]] = None,  # user override
        use_adapter: Optional[bool] = None,           # user override
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> "RepVitDINO":
        if model_name not in REPVIT_DINO_MODELS:
            raise ValueError(f"Unknown model_name: {model_name}. Add it to REPVIT_DINO_MODELS.")

        cfg = REPVIT_DINO_MODELS[model_name]

        final_feature_indices = feature_indices if feature_indices is not None else cfg.get("feature_indices", [-1])
        final_use_adapter = use_adapter if use_adapter is not None else cfg.get("use_adapter", True)

        model = cls(
            backbone_name=cfg["backbone_name"],
            adapter_dim=cfg["adapter_dim"],
            use_adapter=final_use_adapter,
            feature_indices=final_feature_indices,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Supports either raw state_dict or wrapped checkpoint dict
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=strict)

        return model

    
    def forward(self, x):
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor

        Returns:
            Tensor by default (single selected feature), or list of tensors if
            multiple feature indices are requested.
        """
        if self.feature_indices is None:
            selected_features = [self.backbone.forward_features(x).contiguous()]
        else:
            selected_features = self.backbone.forward_intermediates(
                x,
                indices=self.feature_indices,
                intermediates_only=True,
            )
            selected_features = [feat.contiguous() for feat in selected_features]

        if self.use_adapter and len(selected_features) > 0:
            selected_features[-1] = self.adapter(selected_features[-1])

        if len(selected_features) == 1:
            return selected_features[0]
        return selected_features

