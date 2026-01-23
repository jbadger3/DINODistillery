"""
RepVit student models
"""

from torch import nn
import timm



class RepVit(nn.Module):
    """
    Wrapper class to extract features from pretrained DINOv3 ConvNeXt models.
    """
    
    def __init__(self, model_name: str, pretrained: bool = True, out_feature_indexes: list[int] = None):
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
        super(RepVit, self).__init__()

        
        self.model_name = model_name
        self.out_feature_indexes = out_feature_indexes
        
        # Get the timm model name and load the model
        timm_model_name = self._timm_model_name(model_name)
        self.encoder = timm.create_model(timm_model_name, pretrained=pretrained)
        
        
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