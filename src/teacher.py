import torch 
from torch import nn
from typing import Optional, Union, List


class Teacher(nn.Module):
    def __init__(self, model: nn.Module, is_vit: bool = False, out_feature_indexes: Optional[List[int]] = None,):
        """
        Wrapper class for a teacher model.
        
        Args:
            model: The neural network model to be used as a teacher.
            is_vit: Boolean flag indicating if the model is a Vision Transformer (ViT).
        """
        super(Teacher, self).__init__()
        self.model = model
        self.is_vit = is_vit
        self.out_feature_indexes = out_feature_indexes
        self._freeze_model()
        
    def _freeze_model(self):
        """
        Freeze the parameters of the model to prevent them from being updated during training.
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def train(self, mode: bool = True):
        """
        Override train() to keep the teacher model always in eval mode.
        
        Args:
            mode: If True, set to training mode. If False, set to eval mode.
        """
        # Set the Teacher wrapper itself to the requested mode
        super().train(mode)
        
        # But always keep the frozen teacher model in eval mode
        self.model.eval()
        
        return self
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the teacher model.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor(s) from the teacher model.
        """
        return self.model.forward(x)