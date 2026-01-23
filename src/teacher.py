import torch 
from torch import nn
from typing import List, Union

class Teacher(nn.Module):
    def __init__(self, model: nn.Module, teacher_channels: List[int] = None, student_channels: List[int] = None):
        """
        Wrapper class for a teacher model with optional channel adaptation.
        
        Args:
            model: The neural network model to be used as a teacher.
            teacher_channels: List of channel dimensions for each feature level from teacher.
            student_channels: List of channel dimensions for each feature level from student.
                             If channels don't match, 1x1 convolutions are added to adapt.
        """
        super(Teacher, self).__init__()
        self.model = model
        self._freeze_model()
        # Create channel adapters if needed
        self.channel_adapters = nn.ModuleList()
        if teacher_channels is not None and student_channels is not None:
            if len(teacher_channels) != len(student_channels):
                raise ValueError(f"Length mismatch: teacher has {len(teacher_channels)} features, "
                               f"student has {len(student_channels)} features")
            
            for t_ch, s_ch in zip(teacher_channels, student_channels):
                if t_ch != s_ch:
                    # Add 1x1 convolution to match channels
                    adapter = nn.Conv2d(t_ch, s_ch, kernel_size=1, stride=1, padding=0, bias=False)
                    self.channel_adapters.append(adapter)
                else:
                    # No adaptation needed
                    self.channel_adapters.append(nn.Identity())
        
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
        Only the channel adapters should be trainable.
        
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
        Forward pass through the teacher model with channel adaptation.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor(s) from the teacher model, with channels adapted to match student if needed.
        """
        features = self.model(x)
        
        # If no adapters, return as-is
        if len(self.channel_adapters) == 0:
            return features
        
        # Apply channel adapters
        if isinstance(features, (list, tuple)):
            # Multiple features
            adapted_features = []
            for feat, adapter in zip(features, self.channel_adapters):
                adapted_features.append(adapter(feat))
            return adapted_features
        else:
            # Single feature
            return self.channel_adapters[0](features)