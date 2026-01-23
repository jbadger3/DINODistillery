"""
General script to extract stage information from model families in timm.
This will help create a dictionary similar to backbone_registry.py for any model family.
"""

import timm
import torch
import argparse
import os

def analyze_model(model_name, model_family):
    """
    Load a model and extract stage information.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Create model
        model = timm.create_model(model_name, pretrained=False, features_only=True)
        
        # Get feature info
        feature_info = model.feature_info
        
        print(f"\nFeature Info Channels: {feature_info.channels()}")
        print(f"Feature Info Reduction: {feature_info.reduction()}")
        
        # Create stages list similar to backbone_registry
        stages = []
        for i, (channels, reduction) in enumerate(zip(feature_info.channels(), feature_info.reduction())):
            stages.append({
                'channels': channels,
                'stride': reduction
            })
        
        print(f"\nStages for dictionary:")
        print(f"    'stages': [")
        for stage in stages:
            print(f"        {stage},")
        print(f"    ]")
        
        # Test with a sample input
        x = torch.randn(1, 3, 224, 224)
        features = model(x)
        
        print(f"\nOutput feature shapes:")
        for i, feat in enumerate(features):
            print(f"  Stage {i}: {feat.shape}")
        
        return {
            'model_name': model_name,
            'backbone_type': model_family,
            'stages': stages
        }
        
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract stage information from timm model families')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model family to search for (e.g., repvit, convnext, efficientnet)')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Directory to save the output file (default: current directory)')
    args = parser.parse_args()
    
    model_family = args.model
    output_dir = args.output_dir
    
    # Search for all models matching the pattern
    all_base_models = timm.list_models(f'{model_family}*')
    
    if not all_base_models:
        print(f"No models found matching pattern: {model_family}*")
        return
    
    print(f"Found {len(all_base_models)} base models matching '{model_family}':")
    for model_name in all_base_models[:10]:  # Show first 10
        print(f"  - {model_name}")
    if len(all_base_models) > 10:
        print(f"  ... and {len(all_base_models) - 10} more")
    
    # Get pretrained versions
    all_pretrained = timm.list_pretrained(f'{model_family}*')
    
    print(f"\nFound {len(all_pretrained)} pretrained models:")
    for model_name in all_pretrained[:10]:  # Show first 10
        print(f"  - {model_name}")
    if len(all_pretrained) > 10:
        print(f"  ... and {len(all_pretrained) - 10} more")
    
    # Use pretrained models if available, otherwise use base models
    models_to_analyze = all_pretrained if all_pretrained else all_base_models
    
    # Analyze each model - use base name as key since architecture is the same
    model_dict = {}
    analyzed_base_models = set()
    
    for model_name in models_to_analyze:
        # Extract base model name (e.g., 'repvit_m1_1' from 'repvit_m1_1.dist_300e_in1k')
        base_name = model_name.split('.')[0]
        key_name = base_name
        
        # Only analyze each base architecture once (they have same structure)
        if base_name not in analyzed_base_models:
            result = analyze_model(model_name, model_family)
            if result:
                analyzed_base_models.add(base_name)
                model_dict[key_name] = result
    
    # Print final dictionary
    print(f"\n\n{'='*80}")
    print(f"FINAL DICTIONARY FOR {model_family.upper()}_MODELS:")
    print(f"{'='*80}\n")
    
    output_lines = []
    output_lines.append(f"{model_family.upper()}_MODELS = {{")
    for key, value in model_dict.items():
        print(f"    '{key}': {{")
        print(f"        'model_name': '{value['model_name']}',")
        print(f"        'backbone_type': '{value['backbone_type']}',")
        print(f"        'stages': [")
        
        output_lines.append(f"    '{key}': {{")
        output_lines.append(f"        'model_name': '{value['model_name']}',")
        output_lines.append(f"        'backbone_type': '{value['backbone_type']}',")
        output_lines.append(f"        'stages': [")
        
        for stage in value['stages']:
            print(f"            {stage},")
            output_lines.append(f"            {stage},")
        
        print(f"        ]")
        print(f"    }},")
        output_lines.append(f"        ]")
        output_lines.append(f"    }},")
    
    print("}")
    output_lines.append("}")
    
    #s.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_family}_registry.py")
    output_file = f"{model_family}_registry.py"
    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write(f'{model_family.upper()} model configurations extracted from timm.\n')
        f.write('"""\n\n')
        f.write('\n'.join(output_lines))
        f.write('\n')
    
    print(f"\n{'='*80}")
    print(f"Dictionary saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
