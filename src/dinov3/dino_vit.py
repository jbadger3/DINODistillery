"""
Features extraction wrapper for DINOv3 ViT models using timm.

This module provides a wrapper class that extracts features from DINOv3 ViT models
using the timm library.
"""

import torch
from torch import nn
import timm
from .backbone_registry import VIT_MODELS, VIT_MODELS_QKVB


class DINOViT(nn.Module):
    """
    Wrapper class to extract features from pretrained DINOv3 ViT models using timm.
    """
    
    def __init__(self, model_name: str, pretrained: bool = True, out_feature_indexes: list[int] = None, freeze: bool = True):
        """
        Initialize the ViT features model.
        
        Args:
            model_name: Name of ViT model to load (must be from VIT_MODELS or VIT_MODELS_QKVB)
            pretrained: Whether to load pretrained weights (default: True)
            out_feature_indexes: List of layer indices to extract features from
            freeze: Whether to freeze backbone parameters (default: True)
        
        Raises:
            ValueError: If model_name is not a valid ViT model
        """
        super(DINOViT, self).__init__()
        
        # Validate that model_name is a ViT model
        self._validate_model_name(model_name)
        
        self.model_name = model_name
        self.out_feature_indexes = out_feature_indexes
        self.freeze = freeze
        
        # Load the model from timm
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        
        # Freeze backbone parameters if specified
        if self.freeze:
            self._freeze_backbone()
            self.encoder.eval()
    
    def _validate_model_name(self, model_name: str):
        """
        Validate that the model name is a supported ViT model.
        
        Args:
            model_name: Name of the model to validate
            
        Raises:
            ValueError: If model_name is not in the supported ViT models list
        """
        valid_vit_models = [model_dict['model_name'] for model_dict in VIT_MODELS.values()] + [model_dict['model_name'] for model_dict in VIT_MODELS_QKVB.values()]
        
        if model_name not in valid_vit_models:
            raise ValueError(
                f"Invalid model name '{model_name}'. "
                f"Must be one of the supported ViT models: {valid_vit_models}"
            )
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"ViT backbone parameters frozen: {sum(p.numel() for p in self.encoder.parameters())} parameters")
    
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
            print(f"ViT backbone kept in eval mode (frozen={self.freeze})")
        
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
        else:
            outputs = self.encoder.forward_intermediates(x, indices=self.out_feature_indexes, intermediates_only=True)
        return outputs