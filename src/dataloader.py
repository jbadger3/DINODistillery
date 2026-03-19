"""
DataLoaders for various datasets including SA-1B (Segment Anything 1 Billion).
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Union
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
        augmentation_config: Optional[dict] = None,
        student_transform: Optional[Callable] = None,
        teacher_transform: Optional[Callable] = None,
        student_target_size: Optional[int] = None,
        teacher_target_size: Optional[int] = None,
        return_student_teacher: bool = False,
        sync_student_teacher_augs: bool = True
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
            student_transform: Optional student transform for dual-view mode
            teacher_transform: Optional teacher transform for dual-view mode
            student_target_size: Student target size for dual-view mode
            teacher_target_size: Teacher target size for dual-view mode
            return_student_teacher: If True, return (student_image, teacher_image)
            sync_student_teacher_augs: If True, share same random aug decisions
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_size = val_size
        self.transform = transform
        self.return_path = return_path
        self.limit = limit
        self.augmentation_config = augmentation_config
        self.image_size = image_size  # Store for dynamic updates
        self.return_student_teacher = return_student_teacher
        self.sync_student_teacher_augs = sync_student_teacher_augs

        self.student_target_size = student_target_size if student_target_size is not None else image_size
        self.teacher_target_size = teacher_target_size if teacher_target_size is not None else image_size
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        
        # Set up transforms
        if self.return_student_teacher:
            if self.student_transform is None:
                self.student_transform = self._create_transform(self.student_target_size)
            if self.teacher_transform is None:
                self.teacher_transform = self._create_transform(self.teacher_target_size)
        else:
            # Backward-compatible single-view behavior
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

                # Color Jitter
                color_jitter_config = aug_config.get('color_jitter', {})
                if color_jitter_config.get('enabled', False):
                    p = color_jitter_config.get('p', 0.8)
                    brightness = color_jitter_config.get('brightness', 0.8)
                    contrast = color_jitter_config.get('contrast', 0.8)
                    saturation = color_jitter_config.get('saturation', 0.8)
                    hue = color_jitter_config.get('hue', 0.2)

                    transform_list.append(
                        transforms.RandomApply(
                            [
                                transforms.ColorJitter(
                                    brightness=brightness,
                                    contrast=contrast,
                                    saturation=saturation,
                                    hue=hue,
                                )
                            ],
                            p=p,
                        )
                    )

                # Gaussian Blur
                blur_config = aug_config.get('gaussian_blur', {})
                if blur_config.get('enabled', False):
                    p = blur_config.get('p', 0.1)
                    kernel_size = blur_config.get('kernel_size', 23)
                    sigma = blur_config.get('sigma', [0.1, 2.0])

                    if isinstance(kernel_size, list):
                        kernel_size = tuple(kernel_size)
                    if isinstance(sigma, list):
                        sigma = tuple(sigma)

                    transform_list.append(
                        transforms.RandomApply(
                            [transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)],
                            p=p,
                        )
                    )

                # Solarization
                solarization_config = aug_config.get('solarization', {})
                if solarization_config.get('enabled', False):
                    p = solarization_config.get('p', 0.2)
                    threshold = solarization_config.get('threshold', 128)
                    transform_list.append(
                        transforms.RandomApply(
                            [transforms.RandomSolarize(threshold=threshold, p=1.0)],
                            p=p,
                        )
                    )
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

    def update_image_size(self, image_size: int):
        """Update single-view transform image size."""
        self.image_size = image_size
        self.transform = self._create_transform(image_size)

    def update_dual_image_sizes(self, student_target_size: int, teacher_target_size: int):
        """Update dual-view transforms with potentially different sizes."""
        self.student_target_size = student_target_size
        self.teacher_target_size = teacher_target_size
        self.student_transform = self._create_transform(student_target_size)
        self.teacher_transform = self._create_transform(teacher_target_size)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
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
        
        if self.return_student_teacher:
            if self.sync_student_teacher_augs and self.split == 'train':
                # Ensure random crop/flip are identical across student and teacher views
                torch_state = torch.get_rng_state()
                random_state = random.getstate()

                student_image = self.student_transform(image.copy())

                torch.set_rng_state(torch_state)
                random.setstate(random_state)
                teacher_image = self.teacher_transform(image.copy())
            else:
                student_image = self.student_transform(image.copy())
                teacher_image = self.teacher_transform(image.copy())

            if self.return_path:
                return student_image, teacher_image, str(img_path)
            return student_image, teacher_image

        # Apply transforms (single-view mode)
        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, str(img_path)
        return image


def _get_initial_image_size(image_size_config):
    """Get initial image size from int or epoch->size dict."""
    if isinstance(image_size_config, dict):
        return image_size_config[min(image_size_config.keys())]
    return image_size_config


def _scale_image_size(image_size: int, resize_factor: float) -> int:
    """Scale image size by factor and clamp to at least 1 pixel."""
    return max(1, int(round(float(image_size) * float(resize_factor))))


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

    # Normalize augmentation config (supports bool or dict)
    augmentation_config = data_cfg.get('augmentation')
    if isinstance(augmentation_config, bool):
        augmentation_config = {'enabled': augmentation_config}
    
    # Base size from training config (supports progressive schedule from training.image_size)
    image_size_config = training_cfg.get('image_size', 224)
    image_size = _get_initial_image_size(image_size_config)

    # Dual-view settings from data config
    dual_views = data_cfg.get('dual_views', False)
    student_resize_factor = data_cfg.get('student_resize_factor', 1.0)
    teacher_resize_factor = data_cfg.get('teacher_resize_factor', 1.0)
    student_target_size = _scale_image_size(image_size, student_resize_factor)
    teacher_target_size = _scale_image_size(image_size, teacher_resize_factor)
    sync_student_teacher_augs = data_cfg.get('sync_student_teacher_augs', True)
    
    return create_sa1b_dataloaders(
        root_dir=data_cfg.get('root_dir', 'datasets/sa-1b'),
        batch_size=training_cfg.get('batch_size', 32),
        val_size=data_cfg.get('val_size', 10000),
        image_size=image_size,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=True,
        augmentation_config=augmentation_config,
        train_limit=data_cfg.get('train_limit'),
        val_limit=data_cfg.get('val_limit'),
        return_student_teacher=dual_views,
        student_target_size=student_target_size,
        teacher_target_size=teacher_target_size,
        student_resize_factor=student_resize_factor,
        teacher_resize_factor=teacher_resize_factor,
        sync_student_teacher_augs=sync_student_teacher_augs
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
    val_limit: Optional[int] = None,
    return_student_teacher: bool = False,
    student_target_size: Optional[int] = None,
    teacher_target_size: Optional[int] = None,
    student_resize_factor: float = 1.0,
    teacher_resize_factor: float = 1.0,
    sync_student_teacher_augs: bool = True
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
        return_student_teacher: If True, dataset returns student and teacher views
        student_target_size: Optional student target size for dual-view mode
        teacher_target_size: Optional teacher target size for dual-view mode
        student_resize_factor: Student view size multiplier applied to base image_size
        teacher_resize_factor: Teacher view size multiplier applied to base image_size
        sync_student_teacher_augs: If True, sync train-time random augs across views
        
    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    if return_student_teacher:
        student_target_size = student_target_size if student_target_size is not None else _scale_image_size(image_size, student_resize_factor)
        teacher_target_size = teacher_target_size if teacher_target_size is not None else _scale_image_size(image_size, teacher_resize_factor)

    train_dataset = SA1BDataset(
        root_dir=root_dir,
        split='train',
        val_size=val_size,
        transform=train_transform,
        image_size=image_size,
        augmentation_config=augmentation_config,
        limit=train_limit,
        student_transform=None,
        teacher_transform=None,
        student_target_size=student_target_size,
        teacher_target_size=teacher_target_size,
        return_student_teacher=return_student_teacher,
        sync_student_teacher_augs=sync_student_teacher_augs
    )
    
    val_dataset = SA1BDataset(
        root_dir=root_dir,
        split='val',
        val_size=val_size,
        transform=val_transform,
        image_size=image_size,
        augmentation_config=None,  # No augmentation for validation
        limit=val_limit,
        student_transform=None,
        teacher_transform=None,
        student_target_size=student_target_size,
        teacher_target_size=teacher_target_size,
        return_student_teacher=return_student_teacher,
        sync_student_teacher_augs=False
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
