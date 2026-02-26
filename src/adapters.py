import torch
from torch import nn


class BasicAdapter(nn.Module):
    """
    Basic adapter for student feature projection.
    Simple 1x1 convolution to match channel dimensions.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projector = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        
        # Initialization
        nn.init.kaiming_normal_(self.projector.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        """
        Forward pass through basic adapter.
        
        Args:
            x: Student feature tensor
            
        Returns:
            Projected features matching teacher dimensions
        """
        return self.projector(x)


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter for student feature projection.
    Projects student features to match teacher dimensions using a bottleneck architecture.
    Architecture: input_dim -> bottleneck_dim -> output_dim
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
       
        bottleneck_dim = output_dim // 2
        
        self.projector = nn.Sequential(
            nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, output_dim, kernel_size=1, bias=False)
        )
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        """
        Forward pass through bottleneck adapter.
        
        Args:
            x: Student feature tensor
            
        Returns:
            Projected features matching teacher dimensions
        """
        return self.projector(x)


# Adapter registry
ADAPTER_REGISTRY = {
    'basic': BasicAdapter,
    'bottleneck': BottleneckAdapter,
}


def create_adapter(adapter_type: str, input_dim: int, output_dim: int) -> nn.Module:
    """
    Factory function to create an adapter.
    
    Args:
        adapter_type: Type of adapter ('basic', 'bottleneck')
        input_dim: Input channel dimension
        output_dim: Output channel dimension
        
    Returns:
        Adapter module
        
    Raises:
        ValueError: If adapter_type is not recognized
    """
    if adapter_type not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter type: {adapter_type}. "
                        f"Available types: {list(ADAPTER_REGISTRY.keys())}")
    
    return ADAPTER_REGISTRY[adapter_type](input_dim, output_dim)
