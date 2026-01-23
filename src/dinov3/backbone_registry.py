"""
Model names and metadata for DINOv3 models from the timm library.

ViT models include:
- model_name: The timm model identifier
- backbone_type: 'vit' for Vision Transformer models
- num_layers: Number of transformer layers in the model
- embed_dim: Embedding dimension (feature vector size)

ConvNeXt models include:
- model_name: The timm model identifier  
- backbone_type: 'convnext' for ConvNeXt models
- stages: List of stage configurations with channels and stride information
"""

VIT_MODELS = {
    'vit_small': {
        'model_name': 'vit_small_patch16_dinov3',
        'backbone_type': 'vit',
        'num_layers': 12,
        'embed_dim': 384
    },
    'vit_small_plus': {
        'model_name': 'vit_small_plus_patch16_dinov3',
        'backbone_type': 'vit',
        'num_layers': 12,
        'embed_dim': 384
    },
    'vit_base': {
        'model_name': 'vit_base_patch16_dinov3',
        'backbone_type': 'vit',
        'num_layers': 12,
        'embed_dim': 768
    },
    'vit_large': {
        'model_name': 'vit_large_patch16_dinov3',
        'backbone_type': 'vit',
        'num_layers': 24,
        'embed_dim': 1024
    },
    'vit_huge_plus': {
        'model_name': 'vit_huge_plus_patch16_dinov3',
        'backbone_type': 'vit',
        'num_layers': 32,
        'embed_dim': 1280
    },
    'vit_7b': {
        'model_name': 'vit_7b_patch16_dinov3',
        'backbone_type': 'vit',
        'num_layers': 40,
        'embed_dim': 4096
    },
}

VIT_MODELS_QKVB = {
    'vit_small_qkvb': {
        'model_name': 'vit_small_patch16_dinov3_qkvb',
        'backbone_type': 'vit',
        'num_layers': 12,
        'embed_dim': 384
    },
    'vit_small_plus_qkvb': {
        'model_name': 'vit_small_plus_patch16_dinov3_qkvb',
        'backbone_type': 'vit',
        'num_layers': 12,
        'embed_dim': 384
    },
    'vit_base_qkvb': {
        'model_name': 'vit_base_patch16_dinov3_qkvb',
        'backbone_type': 'vit',
        'num_layers': 12,
        'embed_dim': 768
    },
    'vit_large_qkvb': {
        'model_name': 'vit_large_patch16_dinov3_qkvb',
        'backbone_type': 'vit',
        'num_layers': 24,
        'embed_dim': 1024
    },
    'vit_huge_plus_qkvb': {
        'model_name': 'vit_huge_plus_patch16_dinov3_qkvb',
        'backbone_type': 'vit',
        'num_layers': 32,
        'embed_dim': 1280
    },
}

CONVNEXT_MODELS = {
    'convnext_tiny': {
        'model_name': 'convnext_tiny.dinov3_lvd1689m',
        'backbone_type': 'convnext',
        'stages': [
            {'channels': 96, 'stride': 4},
            {'channels': 192, 'stride': 8},
            {'channels': 384, 'stride': 16},
            {'channels': 768, 'stride': 32},
        ]
    },
    'convnext_small': {
        'model_name': 'convnext_small.dinov3_lvd1689m',
        'backbone_type': 'convnext',
        'stages': [
            {'channels': 96, 'stride': 4},
            {'channels': 192, 'stride': 8},
            {'channels': 384, 'stride': 16},
            {'channels': 768, 'stride': 32},
        ]
    },
    'convnext_base': {
        'model_name': 'convnext_base.dinov3_lvd1689m',
        'backbone_type': 'convnext',
        'stages': [
            {'channels': 128, 'stride': 4},
            {'channels': 256, 'stride': 8},
            {'channels': 512, 'stride': 16},
            {'channels': 1024, 'stride': 32},
        ]
    },
    'convnext_large': {
        'model_name': 'convnext_large.dinov3_lvd1689m',
        'backbone_type': 'convnext',
        'stages': [
            {'channels': 192, 'stride': 4},
            {'channels': 384, 'stride': 8},
            {'channels': 768, 'stride': 16},
            {'channels': 1536, 'stride': 32},
        ]
    }
}


