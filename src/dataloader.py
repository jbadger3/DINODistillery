"""
DataLoaders for various datasets including SA-1B (Segment Anything 1 Billion).
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SA1BDataset(Dataset):
    """
    Dataset loader for Segment Anything 1 Billion (SA-1B) dataset.
    
    The SA-1B dataset is organized into parts (SA-1B-Part-XXXXXX directories),
    each containing .jpg images and .json annotations. This loader only uses
    the images.
    
    Strategy for train/val split:
    - The dataset has 1000 parts (000000-000999), each with ~11k images
    - For validation: use first part of Part-000000 (first 10k images)
    - For training: use remaining images from all parts
    
    This ensures:
    1. Fast validation (all images in one directory)
    2. No overlap between train and val
    3. Deterministic split (always same val set)
    """
    
    def __init__(
        self, 
        root_dir: str,
        split: str = 'train',
        val_size: int = 10000,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        return_path: bool = False,
        limit: Optional[int] = None,
        augmentation_config: Optional[dict] = None
    ):
        """
        Initialize SA-1B dataset.
        
        Args:
            root_dir: Root directory containing SA-1B-Part-* directories
            split: 'train' or 'val'
            val_size: Number of images to use for validation (default: 10000)
            transform: Optional transform to apply to images
            image_size: Target image size for default transforms
            return_path: If True, return (image, path) instead of just image
            limit: Optional limit on number of images to load (useful for testing)
            augmentation_config: Optional dict with augmentation settings from config
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_size = val_size
        self.transform = transform
        self.return_path = return_path
        self.limit = limit
        self.augmentation_config = augmentation_config
        self.image_size = image_size  # Store for dynamic updates
        
        # Set up default transforms if none provided
        if self.transform is None:
            self.transform = self._create_transform(self.image_size)
        
        # Collect all image paths
        self.image_paths = self._collect_image_paths()
        
        # Apply limit if specified
        if self.limit is not None and self.limit < len(self.image_paths):
            self.image_paths = self.image_paths[:self.limit]
            print(f"SA-1B {split} dataset initialized with {len(self.image_paths):,} images (limited from larger set)")
        else:
            print(f"SA-1B {split} dataset initialized with {len(self.image_paths):,} images")
    
    def _create_transform(self, image_size: int):
        """
        Create transforms based on split and augmentation config.
        
        For training: Uses configurable augmentations from config
        For validation: Simple resize, center crop, normalize
        
        Note: ToTensor converts PIL Image [0, 255] to float tensor [0.0, 1.0]
              Normalize applies: output = (input - mean) / std
              ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        # ImageNet normalization stats
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if self.split == 'train':
            transform_list = []
            
            # Check if augmentation is enabled
            aug_config = self.augmentation_config
            if aug_config and aug_config.get('enabled', True):
                # Random Resized Crop (always applied if augmentation enabled)
                crop_config = aug_config.get('random_resized_crop', {})
                scale = crop_config.get('scale', [0.5, 1.0])
                ratio = crop_config.get('ratio', [0.75, 1.333])
                transform_list.append(
                    transforms.RandomResizedCrop(
                        image_size, 
                        scale=tuple(scale), 
                        ratio=tuple(ratio)
                    )
                )
                
                # Horizontal Flip
                hflip_config = aug_config.get('horizontal_flip', {})
                if hflip_config.get('enabled', True):
                    p = hflip_config.get('p', 0.5)
                    transform_list.append(transforms.RandomHorizontalFlip(p=p))
            else:
                # Augmentation disabled - use simple random crop
                transform_list.append(transforms.RandomResizedCrop(image_size))
                transform_list.append(transforms.RandomHorizontalFlip())
            
            # Always add ToTensor and Normalize
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            return transforms.Compose(transform_list)
        
        else:  # val
            return transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),  # 256 for 224
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    def _collect_image_paths(self) -> List[Path]:
        """
        Collect image paths based on split strategy.
        
        Val split: First 10k images from Part-000000
        Train split: All other images
        """
        image_paths = []
        
        # Get all part directories sorted
        part_dirs = sorted(self.root_dir.glob('SA-1B-Part-*'))
        
        if len(part_dirs) == 0:
            raise ValueError(f"No SA-1B-Part-* directories found in {self.root_dir}")
        
        if self.split == 'val':
            # Use first val_size images from Part-000000
            first_part = part_dirs[0]
            all_images = sorted(first_part.glob('*.jpg'))
            image_paths = all_images[:self.val_size]
            
            if len(image_paths) < self.val_size:
                print(f"Warning: Only found {len(image_paths)} images, requested {self.val_size}")
        
        else:  # train
            # First part: skip first val_size images
            first_part = part_dirs[0]
            first_part_images = sorted(first_part.glob('*.jpg'))
            image_paths.extend(first_part_images[self.val_size:])
            
            # All other parts: use all images
            for part_dir in part_dirs[1:]:
                part_images = sorted(part_dir.glob('*.jpg'))
                image_paths.extend(part_images)
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get an image from the dataset.
        
        Args:
            idx: Index of the image
            
        Returns:
            If return_path=False: transformed image tensor
            If return_path=True: (image tensor, image path)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, str(img_path)
        else:
            return image


def create_sa1b_dataloaders_from_config(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create SA-1B dataloaders from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'data' and 'training' sections
        
    Returns:
        (train_loader, val_loader)
        
    Example config structure:
        {
            'data': {
                'root_dir': 'datasets/sa-1b',
                'val_size': 10000,
                'augmentation': {...},  # augmentation config dict
                'train_limit': None,  # optional
                'val_limit': None     # optional
            },
            'training': {
                'image_size': 224,  # or dict for progressive training
                'batch_size': 64,
                'num_workers': 4
            }
        }
    """
    data_cfg = config['data']
    training_cfg = config['training']
    
    # Get initial image size (handle both int and dict formats)
    image_size_config = training_cfg.get('image_size', 224)
    if isinstance(image_size_config, dict):
        # Progressive training: use the first (epoch 0) size
        image_size = image_size_config[min(image_size_config.keys())]
    else:
        # Fixed size
        image_size = image_size_config
    
    return create_sa1b_dataloaders(
        root_dir=data_cfg.get('root_dir', 'datasets/sa-1b'),
        batch_size=training_cfg.get('batch_size', 32),
        val_size=data_cfg.get('val_size', 10000),
        image_size=image_size,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=True,
        augmentation_config=data_cfg.get('augmentation'),
        train_limit=data_cfg.get('train_limit'),
        val_limit=data_cfg.get('val_limit')
    )


def create_sa1b_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    val_size: int = 10000,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    augmentation_config: Optional[dict] = None,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for SA-1B dataset.
    
    Args:
        root_dir: Root directory containing SA-1B-Part-* directories
        batch_size: Batch size for dataloaders
        val_size: Number of images for validation set
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation
        augmentation_config: Optional dict with augmentation settings
        train_limit: Optional limit on number of training images (for testing/debugging)
        val_limit: Optional limit on number of validation images (for testing/debugging)
        
    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SA1BDataset(
        root_dir=root_dir,
        split='train',
        val_size=val_size,
        transform=train_transform,
        image_size=image_size,
        augmentation_config=augmentation_config,
        limit=train_limit
    )
    
    val_dataset = SA1BDataset(
        root_dir=root_dir,
        split='val',
        val_size=val_size,
        transform=val_transform,
        image_size=image_size,
        augmentation_config=None,  # No augmentation for validation
        limit=val_limit
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SA-1B DataLoader')
    parser.add_argument('--root_dir', type=str, default='datasets/sa-1b',
                       help='Path to SA-1B dataset root')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of workers')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--train_limit', type=int, default=None,
                       help='Limit number of training images (for testing)')
    parser.add_argument('--val_limit', type=int, default=None,
                       help='Limit number of validation images (for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SA-1B DataLoader Test")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader = create_sa1b_dataloaders(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        val_size=10000,
        image_size=args.image_size,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        val_limit=args.val_limit
    )
    
    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print(f"Total train images: {len(train_loader.dataset):,}")
    print(f"Total val images: {len(val_loader.dataset):,}")
    
    # Test loading a batch
    print("\nLoading test batches...")
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print(f"Train batch shape: {train_batch.shape}")
    print(f"Val batch shape: {val_batch.shape}")
    print(f"Train batch range: [{train_batch.min():.3f}, {train_batch.max():.3f}]")
    
    print("\n✓ DataLoader test completed successfully!")
