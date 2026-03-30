import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread-safe figure rendering
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
import timm

from teacher import Teacher
from student import Student
from gram_loss import GramLoss
from dinov3.backbone_registry import VIT_MODELS, VIT_MODELS_QKVB, CONVNEXT_MODELS
from dinov3.dino_vit import DINOViT
from dinov3.dino_convnext import DINOConvNeXt
from students.repvit.repvit_registry import REPVIT_MODELS
from utils.rgb_maps_for_features import rgb_pca_maps_for_features
import sys


def _scale_image_size(image_size: int, resize_factor: float) -> int:
    """Scale image size by factor and clamp to at least 1 pixel."""
    return max(1, int(round(float(image_size) * float(resize_factor))))


class ImageSizeSGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with restarts aligned to image-size milestones."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        restart_epochs: List[int],
        max_epochs: int,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 1e-3,
        restart_lr_decay: float = 1.0,
        last_epoch: int = -1,
    ):
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.restart_lr_decay = float(restart_lr_decay)

        # Build sorted unique restart points, always including epoch 0
        parsed_restart_epochs = sorted({int(epoch) for epoch in restart_epochs if int(epoch) >= 0})
        if 0 not in parsed_restart_epochs:
            parsed_restart_epochs.insert(0, 0)

        # Ignore restart points beyond max_epochs and append terminal boundary
        parsed_restart_epochs = [epoch for epoch in parsed_restart_epochs if epoch < self.max_epochs]
        self.restart_epochs = parsed_restart_epochs
        self.boundaries = self.restart_epochs + [self.max_epochs]

        super().__init__(optimizer, last_epoch)

    def _find_cycle_bounds(self, epoch: int):
        """Return (cycle_index, cycle_start, cycle_end) containing the given epoch."""
        for idx in range(len(self.boundaries) - 1):
            cycle_start = self.boundaries[idx]
            cycle_end = self.boundaries[idx + 1]
            if cycle_start <= epoch < cycle_end:
                return idx, cycle_start, cycle_end
        return len(self.boundaries) - 2, self.boundaries[-2], self.boundaries[-1]

    def get_lr(self):
        epoch = max(0, int(self.last_epoch))

        # Global linear warmup at training start
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # Use epoch/(warmup_epochs-1) so epoch 0 starts exactly at warmup_start_factor
            # and the final warmup epoch reaches factor 1.0.
            if self.warmup_epochs == 1:
                progress = 1.0
            else:
                progress = float(epoch) / float(self.warmup_epochs - 1)
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            return [base_lr * factor for base_lr in self.base_lrs]

        cycle_index, cycle_start, cycle_end = self._find_cycle_bounds(epoch)

        # For the initial cycle, start cosine decay after warmup completes.
        effective_cycle_start = cycle_start
        if cycle_start == 0 and self.warmup_epochs > 0:
            effective_cycle_start = min(self.warmup_epochs, max(0, cycle_end - 1))

        cycle_length = max(1, int(cycle_end - effective_cycle_start))
        cycle_progress = float(epoch - effective_cycle_start) / float(cycle_length)
        cycle_progress = min(max(cycle_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

        current_cycle_peak_lrs = [
            max(self.min_lr, base_lr * (self.restart_lr_decay ** cycle_index))
            for base_lr in self.base_lrs
        ]

        return [
            self.min_lr + (peak_lr - self.min_lr) * cosine
            for peak_lr in current_cycle_peak_lrs
        ]


def get_model_registry_info(model_name: str, is_teacher: bool) -> Dict[str, Any]:
    """Get registry information for a model.
    
    Args:
        model_name: Name of the model
        is_teacher: Whether this is a teacher model (True) or student model (False)
        
    Returns:
        Dictionary containing model registry information
        
    Raises:
        ValueError: If model not found in registry
    """
    if is_teacher:
        # Check teacher registries
        if model_name in VIT_MODELS:
            return VIT_MODELS[model_name]
        elif model_name in VIT_MODELS_QKVB:
            return VIT_MODELS_QKVB[model_name]
        elif model_name in CONVNEXT_MODELS:
            return CONVNEXT_MODELS[model_name]
        else:
            raise ValueError(f"Teacher model '{model_name}' not found in registry")
    else:
        # Check student registries
        if model_name in REPVIT_MODELS:
            return REPVIT_MODELS[model_name]
        else:
            raise ValueError(f"Student model '{model_name}' not found in registry")


def get_feature_info(model_info: Dict[str, Any], out_feature_indexes: Optional[List[int]]) -> List[Dict[str, int]]:
    """Extract feature information (channels and strides) from model registry.
    
    Args:
        model_info: Registry information for the model
        out_feature_indexes: List of feature indices to extract, or None for last feature only
        
    Returns:
        List of dictionaries containing 'channels' and 'stride' for each feature level
    """
    backbone_type = model_info.get('backbone_type', '')
    
    if backbone_type == 'vit':
        # ViT models output a single feature map with embed_dim channels
        # Stride is typically 16 for patch16 models
        embed_dim = model_info['embed_dim']
        stride = 16  # Default stride for patch16 models
        
        if out_feature_indexes is None:
            # Single output feature
            return [{'channels': embed_dim, 'stride': stride}]
        else:
            # Multiple intermediate features (all have same embed_dim)
            return [{'channels': embed_dim, 'stride': stride} for _ in out_feature_indexes]
    
    elif backbone_type in ['convnext', 'repvit']:
        # ConvNeXt and RepViT models have staged architectures
        stages = model_info['stages']
        
        if out_feature_indexes is None:
            # Only last stage
            return [stages[-1]]
        else:
            # Selected stages
            return [stages[idx] for idx in out_feature_indexes]
    
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def validate_teacher_student_compatibility(
    teacher_features: List[Dict[str, int]], 
    student_features: List[Dict[str, int]],
    teacher_name: str,
    student_name: str
) -> None:
    """Validate that teacher and student models are compatible.
    
    Args:
        teacher_features: List of feature info dicts from teacher
        student_features: List of feature info dicts from student
        teacher_name: Name of teacher model
        student_name: Name of student model
        
    Raises:
        SystemExit: If models are incompatible
    """
    # Check if number of feature levels match
    if len(teacher_features) != len(student_features):
        print(f"ERROR: Number of output features don't match!")
        print(f"Teacher '{teacher_name}' has {len(teacher_features)} output features")
        print(f"Student '{student_name}' has {len(student_features)} output features")
        print(f"Both models must output the same number of feature levels.")
        sys.exit(1)
    

def create_teacher_model(config: Dict[str, Any]) -> Teacher:
    """Create teacher model from config.
    
    Args:
        config: Configuration dictionary containing teacher settings
        
    Returns:
        Teacher wrapper with frozen backbone
    """
    teacher_cfg = config['teacher']
    
    teacher_model_name = teacher_cfg['model']
    teacher_out_feature_indexes = teacher_cfg.get('out_feature_indexes', None)
    
    # Convert empty list to None
    if teacher_out_feature_indexes is not None and len(teacher_out_feature_indexes) == 0:
        teacher_out_feature_indexes = None
    
    # Get registry information
    teacher_info = get_model_registry_info(teacher_model_name, is_teacher=True)
    # Determine model type and create appropriate backbone
    if teacher_model_name in VIT_MODELS or teacher_model_name in VIT_MODELS_QKVB:
        # Get the timm model name for ViT models
        timm_model_name = teacher_info['model_name']
        # Create DINOViT model (freeze=True by default)
        backbone = DINOViT(
            model_name=timm_model_name,
            pretrained=True,
            out_feature_indexes=teacher_out_feature_indexes,
            freeze=True
        )
    elif teacher_model_name in CONVNEXT_MODELS:
        # Create DINOConvNeXt model (freeze=True by default)
        backbone = DINOConvNeXt(
            model_name=teacher_model_name,
            pretrained=True,
            out_feature_indexes=teacher_out_feature_indexes,
            freeze=True
        )
    else:
        raise ValueError(f"Teacher model '{teacher_model_name}' not found in registry. "
                        f"Available models: {list({**VIT_MODELS, **VIT_MODELS_QKVB, **CONVNEXT_MODELS}.keys())}")
    
    # Create Teacher wrapper (no adapters - those are on the student now)
    teacher = Teacher(model=backbone)
    
    return teacher


def create_student_model(config: Dict[str, Any]) -> Student:
    """Create student model from config with channel adaptation.
    
    Args:
        config: Configuration dictionary containing teacher and student settings
        
    Returns:
        Student wrapper with backbone and channel adapters
    """
    teacher_cfg = config['teacher']
    student_cfg = config['student']
    
    teacher_model_name = teacher_cfg['model']
    student_model_name = student_cfg['model']
    
    teacher_out_feature_indexes = teacher_cfg.get('out_feature_indexes', None)
    student_out_feature_indexes = student_cfg.get('out_feature_indexes', None)
    
    # Convert empty list to None
    if teacher_out_feature_indexes is not None and len(teacher_out_feature_indexes) == 0:
        teacher_out_feature_indexes = None
    if student_out_feature_indexes is not None and len(student_out_feature_indexes) == 0:
        student_out_feature_indexes = None
    
    # Get registry information
    teacher_info = get_model_registry_info(teacher_model_name, is_teacher=True)
    student_info = get_model_registry_info(student_model_name, is_teacher=False)
    
    # Get feature information
    teacher_features = get_feature_info(teacher_info, teacher_out_feature_indexes)
    student_features = get_feature_info(student_info, student_out_feature_indexes)
    
    # Validate compatibility (exits if incompatible)
    validate_teacher_student_compatibility(
        teacher_features, student_features,
        teacher_model_name, student_model_name
    )
    
    # Find model in registries
    if student_model_name not in REPVIT_MODELS:
        raise ValueError(f"Student model '{student_model_name}' not found in registry. "
                        f"Available models: {list(REPVIT_MODELS.keys())}")
    
    model_info = REPVIT_MODELS[student_model_name]
    timm_model_name = model_info['model_name']
    pretrained = student_cfg.get('pretrained', False)
    
    # Create student backbone using standard timm model (manual feature extraction in Student wrapper)
    backbone = timm.create_model(
        timm_model_name, 
        pretrained=pretrained,
    )
    
    # Get adapter type from config
    adapter_type = student_cfg.get('adapter_type', 'bottleneck')
    
    # Wrap student backbone with channel adapters
    student = Student(
        model=backbone,
        teacher_channels=[f['channels'] for f in teacher_features],
        student_channels=[f['channels'] for f in student_features],
        out_feature_indexes=student_out_feature_indexes,
        adapter_type=adapter_type
    )
    
    return student


class DistillationLightningModule(L.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        """Initialize distillation module.
        
        Args:
            config: Full configuration dictionary from config file
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Create teacher and student models
        self.teacher = create_teacher_model(config)
        self.student = create_student_model(config)
        
        # Note: Teacher model is automatically kept in eval mode by the Teacher class
        # The Teacher.train() override ensures the frozen backbone stays in eval mode
        
        # Get feature extraction indexes
        self.teacher_feature_indexes = config['teacher'].get('out_feature_indexes', [])
        self.student_feature_indexes = config['student'].get('out_feature_indexes', [])
        
        # Distillation config
        distillation_cfg = config['distillation']
        self.temperature = distillation_cfg.get('temperature', 4.0)

        spatial_matching_cfg = distillation_cfg.get('spatial_matching_mode', {})
        if not isinstance(spatial_matching_cfg, dict):
            raise ValueError(
                "distillation.spatial_matching_mode must be a dict with keys "
                "'name' and 'feature_interpolate_mode'"
            )

        self.spatial_matching_mode = str(
            spatial_matching_cfg.get('name', 'teacher2student')
        ).lower()
        self.feature_interpolate_mode = str(
            spatial_matching_cfg.get('feature_interpolate_mode', 'bilinear')
        ).lower()

        valid_spatial_matching_modes = {'teacher2student', 'student2teacher'}
        if self.spatial_matching_mode not in valid_spatial_matching_modes:
            raise ValueError(
                "Unknown distillation.spatial_matching_mode.name: "
                f"{self.spatial_matching_mode}. "
                f"Expected one of {sorted(valid_spatial_matching_modes)}"
            )

        valid_interpolate_modes = {'bilinear', 'nearest'}
        if self.feature_interpolate_mode not in valid_interpolate_modes:
            raise ValueError(
                "Unknown distillation.spatial_matching_mode.feature_interpolate_mode: "
                f"{self.feature_interpolate_mode}. "
                f"Expected one of {sorted(valid_interpolate_modes)}"
            )
        
        # Parse loss configuration - support both single and multi-loss
        self.loss_configs = self._parse_loss_config(distillation_cfg)
        
        # Stage loss weights (optional)
        stage_loss_weights = distillation_cfg.get('stage_loss_weights', None)
        if stage_loss_weights and len(stage_loss_weights) > 0:
            self.stage_loss_weights = torch.tensor(stage_loss_weights, dtype=torch.float32)
        else:
            self.stage_loss_weights = None

        # Optional Gram loss configuration
        self.gram_loss_config = self._parse_gram_loss_config(distillation_cfg)
        self.gram_loss_enabled = self.gram_loss_config['enabled']
        self.gram_loss_weight = self.gram_loss_config['weight']
        self.gram_loss_img_level = self.gram_loss_config['img_level']
        self.gram_loss_schedule = self.gram_loss_config['epoch_schedule']
        self.gram_loss_fn = None
        if self.gram_loss_enabled:
            self.gram_loss_fn = GramLoss(
                apply_norm=self.gram_loss_config['apply_norm'],
                img_level=self.gram_loss_img_level,
                remove_neg=self.gram_loss_config['remove_neg'],
                remove_only_teacher_neg=self.gram_loss_config['remove_only_teacher_neg'],
            )
        
        # Progressive image size configuration
        training_size_cfg = config['training'].get('image_size', 224)
        data_cfg = config.get('data', {})
        self.dual_views = data_cfg.get('dual_views', False)
        self.student_resize_factor = float(data_cfg.get('student_resize_factor', 1.0))
        self.teacher_resize_factor = float(data_cfg.get('teacher_resize_factor', 1.0))
        self.image_size_schedule = self._parse_image_size_config(training_size_cfg)

    def _to_display_rgb(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized CHW image tensor to display-ready HWC RGB numpy array."""
        image_tensor = image_tensor.detach().float().cpu().clamp(-10, 10)

        # Reverse ImageNet normalization used by dataloader transforms.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
        image_tensor = image_tensor.clamp(0.0, 1.0)
        return image_tensor.permute(1, 2, 0).numpy()

    def _get_tensorboard_experiment(self):
        """Get a TensorBoard-like experiment writer with add_image support."""
        if self.logger is None:
            return None

        experiment = getattr(self.logger, 'experiment', None)
        if experiment is not None and hasattr(experiment, 'add_image'):
            return experiment

        logger_collection = getattr(self.logger, 'loggers', None)
        if logger_collection is not None:
            for logger in logger_collection:
                exp = getattr(logger, 'experiment', None)
                if exp is not None and hasattr(exp, 'add_image'):
                    return exp
        return None

    def _log_validation_feature_visualization(
        self,
        student_images: torch.Tensor,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> None:
        """Render and log PCA RGB visualizations for validation features to TensorBoard."""
        experiment = self._get_tensorboard_experiment()
        if experiment is None:
            return

        max_images_to_log = int(self.config.get('logging', {}).get('val_image_log_max_images', 2))
        max_images_to_log = max(1, min(max_images_to_log, int(student_images.shape[0])))
        feature_indices = self.student_feature_indexes if self.student_feature_indexes else list(range(len(student_features)))

        for feat_list_idx, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # Match spatial size for side-by-side visualization consistency.
            s_feat, t_feat = self._match_feature_spatial_shapes(s_feat, t_feat)
            student_rgb_maps, teacher_rgb_maps = rgb_pca_maps_for_features(s_feat, t_feat)
            student_rgb_maps = student_rgb_maps.detach().cpu()
            teacher_rgb_maps = teacher_rgb_maps.detach().cpu()

            layer_idx = feature_indices[feat_list_idx] if feat_list_idx < len(feature_indices) else feat_list_idx
            for img_idx in range(max_images_to_log):
                original_rgb = self._to_display_rgb(student_images[img_idx])
                teacher_rgb = teacher_rgb_maps[img_idx].numpy()
                student_rgb = student_rgb_maps[img_idx].numpy()

                fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                axes[0].imshow(original_rgb)
                axes[0].set_title(f'Original - Epoch {self.current_epoch}')
                axes[0].axis('off')

                axes[1].imshow(teacher_rgb)
                axes[1].set_title(f'Teacher PCA RGB - Epoch {self.current_epoch}')
                axes[1].axis('off')

                axes[2].imshow(student_rgb)
                axes[2].set_title(f'Student PCA RGB - Epoch {self.current_epoch}')
                axes[2].axis('off')

                fig.suptitle(
                    f'Validation Feature Visualization | Epoch {self.current_epoch} | Layer {layer_idx} | Sample {img_idx}',
                    fontsize=11,
                )
                fig.tight_layout()

                fig.canvas.draw()

                experiment.add_figure(
                    f'val_images/layer_{layer_idx}/sample_{img_idx}',
                    fig,
                    global_step=self.global_step,
                )
                plt.close(fig)
    
    def _parse_loss_config(self, distillation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse loss configuration from distillation config.
        
        Args:
            distillation_config: Distillation section of config
            
        Returns:
            List of loss configurations with normalized weights
        """
        losses = distillation_config.get('losses', [])
        
        if not losses or len(losses) == 0:
            raise ValueError("No losses specified in distillation config. "
                           "Please provide a 'losses' list with at least one loss configuration.")
        
        # Normalize weights
        total_weight = sum(loss['weight'] for loss in losses)
        normalized_losses = []
        for loss in losses:
            temperature = float(loss.get('temperature', 1.0))
            if temperature <= 0:
                raise ValueError(
                    f"distillation.losses[{loss.get('type', 'unknown')}].temperature must be > 0"
                )

            temperature_schedule = None
            if loss['type'] == 'exp_cosine':
                temperature_schedule = loss.get('temperature_schedule')
                if temperature_schedule is not None:
                    if not isinstance(temperature_schedule, dict):
                        raise ValueError(
                            "distillation.losses[exp_cosine].temperature_schedule must be a dict "
                            "mapping epoch -> temperature"
                        )

                    temperature_schedule = {
                        int(epoch): float(schedule_temperature)
                        for epoch, schedule_temperature in temperature_schedule.items()
                    }
                    temperature_schedule = dict(sorted(temperature_schedule.items()))

                    for epoch, schedule_temperature in temperature_schedule.items():
                        if schedule_temperature <= 0:
                            raise ValueError(
                                "distillation.losses[exp_cosine].temperature_schedule values "
                                f"must be > 0, got {schedule_temperature} at epoch {epoch}"
                            )

            normalized_losses.append({
                'type': loss['type'],
                'weight': loss['weight'] / total_weight,
                'temperature': temperature,
                'temperature_schedule': temperature_schedule,
            })
        return normalized_losses

    def _get_loss_temperature(self, loss_cfg: Dict[str, Any], epoch: int) -> float:
        """Resolve the active loss temperature, supporting optional epoch schedules."""
        temperature_schedule = loss_cfg.get('temperature_schedule')
        if not temperature_schedule:
            return float(loss_cfg.get('temperature', 1.0))

        applicable_epochs = [scheduled_epoch for scheduled_epoch in temperature_schedule.keys() if scheduled_epoch <= epoch]
        if not applicable_epochs:
            return float(loss_cfg.get('temperature', 1.0))

        epoch_key = max(applicable_epochs)
        return float(temperature_schedule[epoch_key])

    def _resize_teacher_to_student(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """Resize teacher feature map to match student spatial dimensions."""
        if teacher_feat.shape[2:] == student_feat.shape[2:]:
            return teacher_feat

        if self.feature_interpolate_mode == 'nearest':
            return F.interpolate(
                teacher_feat,
                size=student_feat.shape[2:],
                mode='nearest',
            )

        return F.interpolate(
            teacher_feat,
            size=student_feat.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

    def _resize_student_to_teacher(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Resize student feature map to match teacher spatial dimensions using average pooling."""
        if student_feat.shape[2:] == teacher_feat.shape[2:]:
            return student_feat

        s_h, s_w = student_feat.shape[2], student_feat.shape[3]
        t_h, t_w = teacher_feat.shape[2], teacher_feat.shape[3]

        # Average pooling only downsamples. Use adaptive pooling for exact target size.
        if t_h > s_h or t_w > s_w:
            raise ValueError(
                "Cannot upsample student features with average pooling: "
                f"student={student_feat.shape[2:]}, teacher={teacher_feat.shape[2:]}"
            )

        if s_h % t_h == 0 and s_w % t_w == 0:
            kernel_h = s_h // t_h
            kernel_w = s_w // t_w
            return F.avg_pool2d(
                student_feat,
                kernel_size=(kernel_h, kernel_w),
                stride=(kernel_h, kernel_w),
            )

        return F.adaptive_avg_pool2d(student_feat, output_size=teacher_feat.shape[2:])

    def _match_feature_spatial_shapes(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return student/teacher features with aligned spatial shape per configured matching mode."""
        if self.spatial_matching_mode == 'teacher2student':
            teacher_feat = self._resize_teacher_to_student(teacher_feat, student_feat)
            return student_feat, teacher_feat

        if self.spatial_matching_mode == 'student2teacher':
            student_feat = self._resize_student_to_teacher(student_feat, teacher_feat)
            return student_feat, teacher_feat

        raise ValueError(f"Unsupported spatial matching mode: {self.spatial_matching_mode}")
    
    

    def _parse_image_size_config(self, image_size_config):
        """Parse image size configuration into a schedule.
        
        Args:
            image_size_config: Either an int (fixed size) or dict mapping epochs to sizes
            
        Returns:
            Dict mapping epochs to image sizes, sorted by epoch
        """
        if isinstance(image_size_config, dict):
            # Already a schedule - just sort by epoch
            normalized_schedule = {
                int(epoch): int(size)
                for epoch, size in image_size_config.items()
            }
            return dict(sorted(normalized_schedule.items()))
        else:
            # Single size - no schedule
            return {0: int(image_size_config)}

    def _parse_gram_loss_config(self, distillation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse optional Gram loss configuration.

        Expected config format:
            distillation:
              gram_loss:
                enabled: false
                weight: 1.0
                img_level: true
                apply_norm: true
                remove_neg: true
                remove_only_teacher_neg: false
                epochs:
                  0: true
                  50: false
                  80: true
        """
        gram_cfg = distillation_config.get('gram_loss', {})

        enabled = bool(gram_cfg.get('enabled', False))
        weight = float(gram_cfg.get('weight', 1.0))
        img_level = bool(gram_cfg.get('img_level', True))
        apply_norm = bool(gram_cfg.get('apply_norm', True))
        remove_neg = bool(gram_cfg.get('remove_neg', True))
        remove_only_teacher_neg = bool(gram_cfg.get('remove_only_teacher_neg', False))

        epoch_schedule_cfg = gram_cfg.get('epochs', {0: True})
        if not isinstance(epoch_schedule_cfg, dict):
            raise ValueError("distillation.gram_loss.epochs must be a dict mapping epoch -> bool")

        epoch_schedule = {
            int(epoch): bool(is_enabled)
            for epoch, is_enabled in epoch_schedule_cfg.items()
        }
        epoch_schedule = dict(sorted(epoch_schedule.items()))

        if enabled and weight < 0:
            raise ValueError("distillation.gram_loss.weight must be >= 0")

        if remove_neg and remove_only_teacher_neg:
            raise ValueError(
                "Only one of distillation.gram_loss.remove_neg or "
                "distillation.gram_loss.remove_only_teacher_neg can be true"
            )

        return {
            'enabled': enabled,
            'weight': weight,
            'img_level': img_level,
            'apply_norm': apply_norm,
            'remove_neg': remove_neg,
            'remove_only_teacher_neg': remove_only_teacher_neg,
            'epoch_schedule': epoch_schedule,
        }

    def _is_gram_loss_active(self, epoch: int) -> bool:
        """Return whether Gram loss should be applied at the given epoch."""
        if not self.gram_loss_enabled:
            return False

        applicable_epochs = [e for e in self.gram_loss_schedule.keys() if e <= epoch]
        if not applicable_epochs:
            return False

        epoch_key = max(applicable_epochs)
        return bool(self.gram_loss_schedule[epoch_key])

    def _prepare_gram_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Convert feature tensors to the format expected by GramLoss."""
        if feat.ndim == 4:
            # [B, C, H, W] -> [B, N, C]
            return feat.flatten(2).transpose(1, 2)
        if feat.ndim in (2, 3):
            return feat
        raise ValueError(f"Unsupported feature shape for Gram loss: {feat.shape}")
    
    def _get_current_image_size(self, epoch: int) -> int:
        """Get the image size for a given epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Image size to use for this epoch
        """
        # Find the largest epoch key that is <= current epoch
        applicable_epochs = [e for e in self.image_size_schedule.keys() if e <= epoch]
        if applicable_epochs:
            epoch_key = max(applicable_epochs)
            return self.image_size_schedule[epoch_key]
        else:
            # Fallback to first size
            return self.image_size_schedule[min(self.image_size_schedule.keys())]

    def _unpack_student_teacher_images(self, batch):
        """Unpack batch to (student_images, teacher_images), supporting single and dual views."""
        if isinstance(batch, (tuple, list)):
            # Dual-view dataset output: (student_images, teacher_images[, path])
            if len(batch) >= 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                return batch[0], batch[1]
            # Single-view dataset output with optional metadata: (images[, path])
            if len(batch) >= 1 and isinstance(batch[0], torch.Tensor):
                return batch[0], batch[0]

        # Raw tensor batch
        return batch, batch

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        current_epoch = self.current_epoch
        target_base_size = self._get_current_image_size(current_epoch)
        target_student_size = _scale_image_size(target_base_size, self.student_resize_factor)
        target_teacher_size = _scale_image_size(target_base_size, self.teacher_resize_factor)
        
        # Check if we need to update image size
        if current_epoch in self.image_size_schedule and current_epoch > 0:
            print(f"\n{'='*60}")
            if self.dual_views:
                print(
                    f"Epoch {current_epoch}: Updating image sizes "
                    f"student={target_student_size}x{target_student_size}, "
                    f"teacher={target_teacher_size}x{target_teacher_size}"
                )
            else:
                print(f"Epoch {current_epoch}: Updating image size to {target_base_size}x{target_base_size}")
            print(f"{'='*60}\n")
            
            # Update transforms in train dataloader
            if self.trainer and self.trainer.train_dataloader:
                train_dataset = self.trainer.train_dataloader.dataset
                if hasattr(train_dataset, 'update_dual_image_sizes') and getattr(train_dataset, 'return_student_teacher', False):
                    train_dataset.update_dual_image_sizes(target_student_size, target_teacher_size)
                elif hasattr(train_dataset, 'update_image_size'):
                    train_dataset.update_image_size(target_base_size)
                
                # Update validation dataloader as well
                if self.trainer.val_dataloaders:
                    val_dataloaders = self.trainer.val_dataloaders
                    if isinstance(val_dataloaders, DataLoader):
                        val_dataloaders = [val_dataloaders]

                    val_dataset = val_dataloaders[0].dataset
                    if hasattr(val_dataset, 'update_dual_image_sizes') and getattr(val_dataset, 'return_student_teacher', False):
                        val_dataset.update_dual_image_sizes(target_student_size, target_teacher_size)
                    elif hasattr(val_dataset, 'update_image_size'):
                        val_dataset.update_image_size(target_base_size)
        
    def forward(self, x):
        """Forward pass through the student model.
        
        Args:
            x: Input tensor
            
        Returns:
            Student output features
        """
        return self.student(x)
    
    def _compute_single_loss(self, s_feat: torch.Tensor, t_feat: torch.Tensor, loss_cfg: Dict[str, Any]) -> torch.Tensor:
        """Compute a single loss between student and teacher feature.
        
        Args:
            s_feat: Student feature tensor [B, C, H, W]
            t_feat: Teacher feature tensor [B, C, H, W]
            loss_cfg: Loss configuration including type and optional temperature
            
        Returns:
            Loss value
        """
        # Match spatial dimensions according to configured matching mode.
        s_feat, t_feat = self._match_feature_spatial_shapes(s_feat, t_feat)

        loss_type = loss_cfg['type']
        temperature = self._get_loss_temperature(loss_cfg, self.current_epoch)

        if loss_type == 'mse':
            s_norm = F.normalize(s_feat, p=2, dim=1, eps=1e-6)
            t_norm = F.normalize(t_feat, p=2, dim=1, eps=1e-6)
            return F.mse_loss(s_norm, t_norm)
        
        elif loss_type == 'cosine':
            # Spatial cosine similarity (preserves spatial structure)
            # Shape: [B, C, H, W]
            B, C, H, W = s_feat.shape
            
            # Reshape to [B, C, H*W] and transpose to [B, H*W, C]
            s_reshaped = s_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            t_reshaped = t_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Normalize features (L2 normalization per spatial location)
            s_norm = F.normalize(s_reshaped, p=2, dim=2, eps=1e-6)  # [B, H*W, C]
            t_norm = F.normalize(t_reshaped, p=2, dim=2, eps=1e-6)  # [B, H*W, C]
            
            # Compute cosine similarity at each spatial location
            # Einstein sum: batch, spatial, channels -> batch, spatial
            cosine_sim = torch.einsum('bpc,bpc->bp', s_norm, t_norm)  # [B, H*W]
            
            # Loss is 1 - mean(cosine_similarity)
            # Average over spatial locations and batch
            # Optional temperature scaling: lower temperatures amplify gradients.
            return (1.0 - cosine_sim.mean()) / temperature
        
        elif loss_type == 'cosine_global':
            # Global pooling version (alternative option)
            s_flat = s_feat.flatten(2).mean(dim=2)  # [B, C]
            t_flat = t_feat.flatten(2).mean(dim=2)  # [B, C]
            return (1.0 - F.cosine_similarity(s_flat, t_flat, dim=1).mean()) / temperature

        elif loss_type == 'exp_cosine':
            # Exponential cosine variant:
            # L = 1 - exp(tau * cos(S, T) - 1)
            s_reshaped = s_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            t_reshaped = t_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

            s_norm = F.normalize(s_reshaped, p=2, dim=2, eps=1e-6)
            t_norm = F.normalize(t_reshaped, p=2, dim=2, eps=1e-6)

            cosine_sim = torch.einsum('bpc,bpc->bp', s_norm, t_norm)  # [B, H*W]
            return 1.0 - torch.exp(temperature * cosine_sim - 1.0).mean()
        
        elif loss_type == 'kl_div':
            # Apply softmax with temperature
            s_soft = F.log_softmax(s_feat.flatten(2) / self.temperature, dim=1)
            t_soft = F.softmax(t_feat.flatten(2) / self.temperature, dim=1)
            return F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.temperature ** 2)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute_distillation_loss(self, student_features: List[torch.Tensor], 
                                  teacher_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute distillation loss between student and teacher features.
        
        Supports multiple loss types with weighted combination.
        
        Args:
            student_features: List of feature tensors from student
            teacher_features: List of feature tensors from teacher
            
        Returns:
            Dictionary containing 'total' loss, individual loss components, and feature losses
        """
        # Handle single tensor outputs (convert to list)
        if not isinstance(student_features, (list, tuple)):
            student_features = [student_features]
        if not isinstance(teacher_features, (list, tuple)):
            teacher_features = [teacher_features]
        
        # Ensure same number of features
        if len(student_features) != len(teacher_features):
            raise ValueError(f"Mismatch in number of features: student has {len(student_features)}, "
                           f"teacher has {len(teacher_features)}")
        
        total_loss = 0.0
        loss_dict = {}
        
        # Initialize loss component tracking for each loss type
        loss_components = {loss_cfg['type']: 0.0 for loss_cfg in self.loss_configs}
        
        for idx, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            feature_loss = 0.0
            
            # Compute each loss type and combine with weights
            for loss_cfg in self.loss_configs:
                loss_type = loss_cfg['type']
                loss_weight = loss_cfg['weight']
                
                # Compute single loss
                single_loss = self._compute_single_loss(s_feat, t_feat, loss_cfg)
                
                # Weight and accumulate
                weighted_loss = single_loss * loss_weight
                feature_loss += weighted_loss
                
                # Track component for logging
                loss_components[loss_type] += single_loss.detach()
            
            # Store individual feature loss (before stage weighting)
            loss_dict[f'feat_{idx}'] = feature_loss.detach()
            
            # Apply stage-specific weight if specified
            if self.stage_loss_weights is not None:
                if idx >= len(self.stage_loss_weights):
                    raise ValueError(f"Stage loss weights list has {len(self.stage_loss_weights)} elements, "
                                   f"but trying to access index {idx}. Ensure stage_loss_weights has "
                                   f"same length as number of output features ({len(student_features)}).")
                weight = self.stage_loss_weights[idx].to(feature_loss.device)
                feature_loss = feature_loss * weight
                loss_dict[f'feat_{idx}_weighted'] = feature_loss.detach()
            
            # Accumulate loss
            total_loss += feature_loss
        
        # Average across features
        total_loss = total_loss / len(student_features)

        # Optional Gram loss with epoch schedule
        gram_active = self._is_gram_loss_active(self.current_epoch)
        loss_dict['gram_active'] = torch.tensor(float(gram_active), device=total_loss.device)
        if gram_active and self.gram_loss_fn is not None and self.gram_loss_weight > 0:
            gram_loss = 0.0
            for s_feat, t_feat in zip(student_features, teacher_features):
                s_feat, t_feat = self._match_feature_spatial_shapes(s_feat, t_feat)
                s_for_gram = self._prepare_gram_features(s_feat)
                t_for_gram = self._prepare_gram_features(t_feat)
                gram_loss = gram_loss + self.gram_loss_fn(
                    s_for_gram,
                    t_for_gram,
                    img_level=self.gram_loss_img_level,
                )
            gram_loss = gram_loss / len(student_features)
            gram_loss_weighted = gram_loss * self.gram_loss_weight
            total_loss = total_loss + gram_loss_weighted

            loss_dict['gram'] = gram_loss.detach()
            loss_dict['gram_weighted'] = gram_loss_weighted.detach()
        
        # Add total to dict
        loss_dict['total'] = total_loss
        
        # Add individual loss components (averaged across features)
        for loss_type, component_loss in loss_components.items():
            loss_dict[f'loss_{loss_type}'] = component_loss / len(student_features)
        
        return loss_dict
    
    def _calculate_angular_spread(self, features: torch.Tensor):
        """
        Measures how 'diverse' the features are on the hypersphere.
        Args:
            features: [B, C, H, W] tensor
        Returns:
            spread_deg: Average angle (in degrees) between random pairs in the batch
        """
        B, C, H, W = features.shape
        # 1. Global Average Pooling to get one vector per image
        # (Or sample specific spatial locations)
        v = torch.mean(features, dim=(2, 3))  # [B, C]
        
        # 2. L2 Normalize to project onto the unit sphere
        v_norm = F.normalize(v, p=2, dim=1)
        
        # 3. Compute all-to-all cosine similarity matrix [B, B]
        cos_sim_matrix = torch.matmul(v_norm, v_norm.t())
        
        # 4. Extract off-diagonal elements (pairs of different images)
        # We want to know how different image A is from image B
        mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
        pair_sims = cos_sim_matrix[mask]
        
        # 5. Convert to degrees for human readability
        # Clamp to avoid acos NaNs at 1.0000001
        angles_rad = torch.acos(pair_sims.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        spread_deg = torch.rad2deg(angles_rad.mean())
        
        return spread_deg
    
    def _compute_gradient_norms(self):
        """Compute gradient norms for student adapters and student stages.
        
        Returns:
            Dictionary containing gradient norms for each component
        """
        grad_norms = {}
        
        # Compute gradient norms for student channel adapters
        for idx, adapter in enumerate(self.student.channel_adapters):
            if not isinstance(adapter, nn.Identity):
                total_norm = 0.0
                num_params = 0
                for p in adapter.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        num_params += 1
                if num_params > 0:
                    total_norm = total_norm ** 0.5
                    grad_norms[f'student_adapter_{idx}'] = total_norm
        
        # Compute gradient norms for student backbone stages
        if hasattr(self.student.model, 'stages'):
            for idx, stage in enumerate(self.student.model.stages):
                total_norm = 0.0
                num_params = 0
                for p in stage.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        num_params += 1
                if num_params > 0:
                    total_norm = total_norm ** 0.5
                    grad_norms[f'student_stage_{idx}'] = total_norm
        
        return grad_norms
    
    def training_step(self, batch, batch_idx):
        """Training step for knowledge distillation.
        
        Args:
            batch: Input batch containing images and optional labels
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        student_images, teacher_images = self._unpack_student_teacher_images(batch)
        
        # Get student features
        student_features = self.student(student_images)
        
        # Get teacher features (no gradient)
        with torch.no_grad():
            teacher_features = self.teacher(teacher_images)
        
        # Compute distillation loss (returns dict)
        loss_dict = self.compute_distillation_loss(student_features, teacher_features)
        
        # Log all metrics
        self.log('train_loss', loss_dict['total'], on_step=True, on_epoch=True, prog_bar=True)
        
        # Log individual feature losses and loss components
        for key, value in loss_dict.items():
            if key != 'total':
                self.log(f'train_{key}_loss' if not key.startswith('loss_') else f'train_{key}', 
                        value, on_step=False, on_epoch=True, prog_bar=False)

        # Log learning rates once per epoch (from the first training batch).
        if batch_idx == 0 and self.trainer is not None and len(self.trainer.optimizers) > 0:
            optimizer = self.trainer.optimizers[0]
            lrs = [group['lr'] for group in optimizer.param_groups]
            if len(lrs) > 0:
                self.log('lr', lrs[0], on_step=False, on_epoch=True, prog_bar=False)
                self.log('lr/min', min(lrs), on_step=False, on_epoch=True, prog_bar=False)
                self.log('lr/max', max(lrs), on_step=False, on_epoch=True, prog_bar=False)
                for group_idx, lr_value in enumerate(lrs):
                    self.log(f'lr/group_{group_idx}', lr_value, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss_dict['total']
    
    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer.step(). Log gradient norms here.
        
        Args:
            optimizer: The optimizer being used
        """
        # Compute and log gradient norms
        grad_norms = self._compute_gradient_norms()
        for name, norm in grad_norms.items():
            self.log(f'grad_norm/{name}', norm, on_step=False, on_epoch=True, prog_bar=False)
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: Input batch containing images and optional labels
            batch_idx: Batch index
        """
        student_images, teacher_images = self._unpack_student_teacher_images(batch)

        with torch.no_grad():
            # Validation does not require autograd; keep the full path graph-free.
            student_features = self.student(student_images)
            teacher_features = self.teacher(teacher_images)

            # Log feature visualizations once per validation epoch.
            if batch_idx == 0:
                student_features_list_for_viz = list(student_features) if isinstance(student_features, (list, tuple)) else [student_features]
                teacher_features_list_for_viz = list(teacher_features) if isinstance(teacher_features, (list, tuple)) else [teacher_features]
                self._log_validation_feature_visualization(
                    student_images=student_images,
                    student_features=student_features_list_for_viz,
                    teacher_features=teacher_features_list_for_viz,
                )

            # Compute distillation loss (returns dict)
            loss_dict = self.compute_distillation_loss(student_features, teacher_features)

            # Log all metrics
            self.log('val_loss', loss_dict['total'], on_step=False, on_epoch=True, prog_bar=True)

            # Log individual feature losses and loss components
            for key, value in loss_dict.items():
                if key != 'total':
                    self.log(f'val_{key}_loss' if not key.startswith('loss_') else f'val_{key}', 
                            value, on_step=False, on_epoch=True, prog_bar=False)

            # Calculate and log angular spread for feature diversity monitoring
            # Handle single tensor outputs (convert to list) - make copies for iteration
            student_features_list = list(student_features) if isinstance(student_features, (list, tuple)) else [student_features]
            teacher_features_list = list(teacher_features) if isinstance(teacher_features, (list, tuple)) else [teacher_features]

            # Get actual feature indices for logging (use list index if indices not specified)
            feature_indices = self.student_feature_indexes if self.student_feature_indexes else list(range(len(student_features_list)))

            for list_idx, (s_feat, t_feat) in enumerate(zip(student_features_list, teacher_features_list)):
                # Use actual layer index for logging
                layer_idx = feature_indices[list_idx] if list_idx < len(feature_indices) else list_idx

                # Calculate angular spread for student features
                student_spread = self._calculate_angular_spread(s_feat)
                self.log(f'val_angular_spread/student_feat_{layer_idx}', student_spread.item(), 
                        on_step=True, on_epoch=True, prog_bar=False)

                # Calculate angular spread for teacher features
                teacher_spread = self._calculate_angular_spread(t_feat)
                self.log(f'val_angular_spread/teacher_feat_{layer_idx}', teacher_spread.item(), 
                        on_step=True, on_epoch=True, prog_bar=False)

                # Calculate the difference (how well student matches teacher diversity)
                spread_diff = torch.abs(student_spread - teacher_spread)
                self.log(f'val_angular_spread/diff_feat_{layer_idx}', spread_diff.item(), 
                        on_step=True, on_epoch=True, prog_bar=False)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer configuration (with optional scheduler)
        """
        optimizer_cfg = self.config['training']['optimizer']
        optimizer_name = optimizer_cfg['name'].lower()
        lr = optimizer_cfg['lr']
        weight_decay = optimizer_cfg['weight_decay']
        llrd_decay = optimizer_cfg.get('llrd_decay', 0.95)
        
        # Create parameter groups with LLRD
        param_groups = []
        
        # 1. Student adapters: High LR with regularization
        param_groups.append({
            'params': self.student.channel_adapters.parameters(),
            'lr': lr * 10,
            'weight_decay': 0.05
        })
        
        # 2. Student backbone with LLRD: Apply layer-wise decay
        # Access student backbone stages if available (timm models typically have stages or blocks)
        if hasattr(self.student.model, 'stages'):
            stages = list(self.student.model.stages)
            num_stages = len(stages)
            
            # Apply LLRD: deepest stage gets base_lr, earlier stages get decayed LR
            for i, stage in enumerate(reversed(stages)):
                stage_lr = lr * (llrd_decay ** i)
                param_groups.append({
                    'params': stage.parameters(),
                    'lr': stage_lr,
                    'weight_decay': weight_decay
                })
            
            # Stem/embedding gets most decay
            if hasattr(self.student.model, 'patch_embed') or hasattr(self.student.model, 'stem'):
                stem_module = getattr(self.student.model, 'patch_embed', None) or getattr(self.student.model, 'stem', None)
                stem_lr = lr * (llrd_decay ** num_stages)
                param_groups.append({
                    'params': stem_module.parameters(),
                    'lr': stem_lr,
                    'weight_decay': weight_decay
                })
        else:
            # Fallback: No stage structure, use all student params with base LR
            param_groups.append({
                'params': self.student.model.parameters(),
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(param_groups)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(param_groups)
        elif optimizer_name == 'sgd':
            momentum = optimizer_cfg.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup learning rate scheduler if configured
        scheduler_cfg = self.config['training'].get('scheduler')
        if scheduler_cfg:
            scheduler = self._create_scheduler(optimizer, scheduler_cfg)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        return optimizer
    
    def _create_scheduler(self, optimizer, scheduler_cfg):
        """Create learning rate scheduler from config.
        
        Args:
            optimizer: The optimizer to schedule
            scheduler_cfg: Scheduler configuration dict
            
        Returns:
            Learning rate scheduler
        """
        scheduler_name = scheduler_cfg['name'].lower()
        max_epochs = self.config['training']['max_epochs']
        
        if scheduler_name == 'cosine':
            warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
            min_lr = scheduler_cfg.get('min_lr', 0.0)
            
            if warmup_epochs > 0:
                # Cosine annealing with warmup
                from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
                
                # Warmup phase
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                
                # Cosine annealing phase
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max_epochs - warmup_epochs,
                    eta_min=min_lr
                )
                
                # Combine them
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                # Just cosine annealing
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max_epochs,
                    eta_min=min_lr
                )
        
        elif scheduler_name in ['sgdr', 'cosine_restarts']:
            warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
            warmup_start_factor = scheduler_cfg.get('warmup_start_factor', 1e-3)
            min_lr = scheduler_cfg.get('min_lr', 0.0)
            restart_lr_decay = scheduler_cfg.get('restart_lr_decay', 1.0)
            restart_epochs = sorted(self.image_size_schedule.keys())

            scheduler = ImageSizeSGDRScheduler(
                optimizer=optimizer,
                restart_epochs=restart_epochs,
                max_epochs=max_epochs,
                min_lr=min_lr,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=warmup_start_factor,
                restart_lr_decay=restart_lr_decay,
            )

        elif scheduler_name == 'step':
            from torch.optim.lr_scheduler import StepLR
            step_size = scheduler_cfg.get('step_size', 30)
            gamma = scheduler_cfg.get('gamma', 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_name == 'polynomial':
            from torch.optim.lr_scheduler import PolynomialLR
            power = scheduler_cfg.get('power', 1.0)
            scheduler = PolynomialLR(optimizer, total_iters=max_epochs, power=power)
        
        elif scheduler_name == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            gamma = scheduler_cfg.get('gamma', 0.95)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        elif scheduler_name == 'constant':
            warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
            warmup_start_factor = scheduler_cfg.get('warmup_start_factor', 1e-3)
            
            if warmup_epochs > 0:
                # Warmup followed by constant learning rate
                from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
                
                # Warmup phase
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                
                # Constant LR phase
                constant_scheduler = ConstantLR(optimizer, factor=1.0)
                
                # Combine them
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, constant_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                # Just constant learning rate (no warmup)
                from torch.optim.lr_scheduler import ConstantLR
                scheduler = ConstantLR(optimizer, factor=1.0)
        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return scheduler
