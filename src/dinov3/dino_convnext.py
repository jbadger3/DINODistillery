"""
Features extraction wrapper for DINOv3 ConvNeXt models.
"""

from torch import nn
import timm
from .backbone_registry import CONVNEXT_MODELS


class DINOConvNeXt(nn.Module):
    """
    Wrapper class to extract features from pretrained DINOv3 ConvNeXt models.
    """
    
    def __init__(self, model_name: str, pretrained: bool = True, out_feature_indexes: list[int] = None, freeze: bool = True):
        """
        Initialize the ConvNeXt features model.
        
        Args:
            model_name: Name of ConvNeXt model to load (must be from CONVNEXT_MODELS)
            pretrained: Whether to load pretrained weights (default: True)
            out_feature_indexes: List of layer indices to extract features from
            freeze: Whether to freeze backbone parameters (default: True)
        
        Raises:
            ValueError: If model_name is not a valid ConvNeXt model
        """
        super(DINOConvNeXt, self).__init__()
        print(model_name)
        # Validate that model_name is a ConvNeXt model
        self._validate_model_name(model_name)
        
        self.model_name = model_name
        self.out_feature_indexes = out_feature_indexes
        self.freeze = freeze
        
        # Get the timm model name and load the model
        timm_model_name = self._timm_model_name(model_name)
        self.encoder = timm.create_model(timm_model_name, pretrained=pretrained)
        
        # Freeze backbone parameters if specified
        if self.freeze:
            self._freeze_backbone()
            self.encoder.eval()
    
    def _validate_model_name(self, model_name: str):
        """
        Validate that the model name is a supported ConvNeXt model.
        
        Args:
            model_name: Name of the model to validate
            
        Raises:
            ValueError: If model_name is not in the supported ConvNeXt models list
        """
        if model_name not in CONVNEXT_MODELS.keys():
            raise ValueError(
                f"Invalid model name '{model_name}'. "
                f"Must be one of the supported ConvNeXt models: {CONVNEXT_MODELS}"
            )
    
    def _timm_model_name(self, model_name: str) -> str:
        """
        Get the timm model name from the CONVNEXT_MODELS dictionary.
        
        Args:
            model_name: Name of the model key in CONVNEXT_MODELS
            
        Returns:
            The corresponding timm model name string
        """
        return CONVNEXT_MODELS[model_name]['model_name']
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"ConvNeXt backbone parameters frozen: {sum(p.numel() for p in self.encoder.parameters())} parameters")
    
    def train(self, mode: bool = True):
        """
        Set the module in training mode, but keep backbone in eval mode if frozen.
        
        Args:
            mode: whether to set training mode (True) or evaluation mode (False).
        
        Returns:
            Module: self
        """
        super().train(mode)
        
        # If backbone is frozen, always keep it in eval mode
        if self.freeze:
            self.encoder.eval()
        
        return self
       
        
    def forward(self, x):
        """
        Forward pass that returns either the features of the last hidden layer or
        a tuple of features from specified layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor from the model backbone
        """
        if self.out_feature_indexes == None:
            outputs = self.encoder.forward_features(x)
            outputs = outputs.contiguous()
        else:
            outputs = self.encoder.forward_intermediates(x, indices=self.out_feature_indexes, intermediates_only=True)
            # Ensure all intermediate features are contiguous for MPS compatibility
            outputs = [feat.contiguous() for feat in outputs]
        return outputs