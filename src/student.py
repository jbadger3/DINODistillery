import torch
from torch import nn
from typing import List, Union
from adapters import create_adapter


class Student(nn.Module):
    def __init__(self, model: nn.Module, teacher_channels: List[int], student_channels: List[int], 
                 adapter_type: str = 'bottleneck'):
        """
        Wrapper class for a student model with channel adaptation.
        
        The Student wrapper applies trainable adapters to the student's output features
        to match the teacher's channel dimensions. This allows the student to learn
        from teacher features with different channel dimensions.
        
        Args:
            model: The neural network model to be used as a student.
            teacher_channels: List of channel dimensions for each feature level from teacher.
            student_channels: List of channel dimensions for each feature level from student.
                             Adapters are added to transform student features to match teacher.
            adapter_type: Type of adapter to use ('basic', 'bottleneck'). Default: 'bottleneck'
        """
        super(Student, self).__init__()
        self.model = model
        
        # Validate input
        if len(teacher_channels) != len(student_channels):
            raise ValueError(f"Length mismatch: teacher has {len(teacher_channels)} features, "
                           f"student has {len(student_channels)} features")
        
        # Create channel adapters to match teacher dimensions
        self.channel_adapters = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch != s_ch:
                # Add adapter: student_channels -> teacher_channels
                adapter = create_adapter(adapter_type, input_dim=s_ch, output_dim=t_ch)
                self.channel_adapters.append(adapter)
            else:
                # No adaptation needed
                self.channel_adapters.append(nn.Identity())
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the student model with channel adaptation.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor(s) from the student model, with channels adapted to match teacher.
        """
        features = self.model(x)
        
        # Apply channel adapters to transform student features to teacher dimensions
        if isinstance(features, (list, tuple)):
            # Multiple features
            adapted_features = []
            for feat, adapter in zip(features, self.channel_adapters):
                adapted_features.append(adapter(feat))
            return adapted_features
        else:
            # Single feature
            return self.channel_adapters[0](features)
