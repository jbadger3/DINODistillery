"""
RepViT model configurations extracted from timm.
Generated automatically by extract_repvit_info.py
"""

REPVIT_MODELS = {
    'repvit_m0_9': {
        'model_name': 'repvit_m0_9.dist_300e_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 48, 'stride': 4},
            {'channels': 96, 'stride': 8},
            {'channels': 192, 'stride': 16},
            {'channels': 384, 'stride': 32},
        ]
    },
    'repvit_m1': {
        'model_name': 'repvit_m1.dist_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 48, 'stride': 4},
            {'channels': 96, 'stride': 8},
            {'channels': 192, 'stride': 16},
            {'channels': 384, 'stride': 32},
        ]
    },
    'repvit_m1_0': {
        'model_name': 'repvit_m1_0.dist_300e_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 56, 'stride': 4},
            {'channels': 112, 'stride': 8},
            {'channels': 224, 'stride': 16},
            {'channels': 448, 'stride': 32},
        ]
    },
    'repvit_m1_1': {
        'model_name': 'repvit_m1_1.dist_300e_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 64, 'stride': 4},
            {'channels': 128, 'stride': 8},
            {'channels': 256, 'stride': 16},
            {'channels': 512, 'stride': 32},
        ]
    },
    'repvit_m1_5': {
        'model_name': 'repvit_m1_5.dist_300e_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 64, 'stride': 4},
            {'channels': 128, 'stride': 8},
            {'channels': 256, 'stride': 16},
            {'channels': 512, 'stride': 32},
        ]
    },
    'repvit_m2': {
        'model_name': 'repvit_m2.dist_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 64, 'stride': 4},
            {'channels': 128, 'stride': 8},
            {'channels': 256, 'stride': 16},
            {'channels': 512, 'stride': 32},
        ]
    },
    'repvit_m2_3': {
        'model_name': 'repvit_m2_3.dist_300e_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 80, 'stride': 4},
            {'channels': 160, 'stride': 8},
            {'channels': 320, 'stride': 16},
            {'channels': 640, 'stride': 32},
        ]
    },
    'repvit_m3': {
        'model_name': 'repvit_m3.dist_in1k',
        'backbone_type': 'repvit',
        'stages': [
            {'channels': 64, 'stride': 4},
            {'channels': 128, 'stride': 8},
            {'channels': 256, 'stride': 16},
            {'channels': 512, 'stride': 32},
        ]
    },
}
