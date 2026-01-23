import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, List, Optional, Any
import timm

from .teacher import Teacher
from .dinov3.backbone_registry import VIT_MODELS, VIT_MODELS_QKVB, CONVNEXT_MODELS
from .dinov3.dino_vit import DINOViT
from .dinov3.dino_convnext import DINOConvNeXt
from .students.repvit.repvit_registry import REPVIT_MODELS
import sys


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
    
    # Check if strides match for each feature level
    for idx, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
        if t_feat['stride'] != s_feat['stride']:
            print(f"ERROR: Feature strides don't match at level {idx}!")
            print(f"Teacher stride: {t_feat['stride']}, Student stride: {s_feat['stride']}")
            print(f"Both models must have matching strides for all feature levels.")
            sys.exit(1)


def create_teacher_model(config: Dict[str, Any]) -> Teacher:
    """Create teacher model from config with channel adaptation if needed.
    
    Args:
        config: Configuration dictionary containing teacher and student settings
        
    Returns:
        Teacher wrapper with backbone and optional channel adapters
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
    
    # Create Teacher wrapper with channel adapters
    teacher = Teacher(
        model=backbone,
        teacher_channels=[f['channels'] for f in teacher_features],
        student_channels=[f['channels'] for f in student_features]
    )
    
    return teacher


def create_student_model(config: Dict[str, Any]) -> nn.Module:
    """Create student model from config.
    
    Args:
        config: Configuration dictionary containing student settings
        
    Returns:
        Student model
    """
    student_cfg = config['student']
    model_name = student_cfg['model']
    pretrained = student_cfg.get('pretrained', False)
    out_feature_indexes = student_cfg.get('out_feature_indexes', None)
    
    # Convert empty list to None
    if out_feature_indexes is not None and len(out_feature_indexes) == 0:
        out_feature_indexes = None
    
    # Find model in registries
    if model_name not in REPVIT_MODELS:
        raise ValueError(f"Student model '{model_name}' not found in registry. "
                        f"Available models: {list(REPVIT_MODELS.keys())}")
    
    model_info = REPVIT_MODELS[model_name]
    timm_model_name = model_info['model_name']
    
    # Create model using timm with features_only mode
    model = timm.create_model(
        timm_model_name, 
        pretrained=pretrained, 
        features_only=True,
        out_indices=out_feature_indexes if out_feature_indexes else None
    )
    
    return model


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
        self.temperature = config['distillation'].get('temperature', 4.0)
        self.alpha = config['distillation'].get('alpha', 0.5)
        self.loss_type = config['distillation'].get('loss_type', 'kl_div')
        
    def forward(self, x):
        """Forward pass through the student model.
        
        Args:
            x: Input tensor
            
        Returns:
            Student output features
        """
        return self.student(x)
    
    def compute_distillation_loss(self, student_features: List[torch.Tensor], 
                                  teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute distillation loss between student and teacher features.
        
        Args:
            student_features: List of feature tensors from student
            teacher_features: List of feature tensors from teacher
            
        Returns:
            Distillation loss
        """
        # Handle single tensor outputs (convert to list)
        if not isinstance(student_features, (list, tuple)):
            student_features = [student_features]
        if not isinstance(teacher_features, (list, tuple)):
            teacher_features = [teacher_features]
        
        # If feature indexes are specified, features are already filtered by the models
        # No need to filter again here
        
        # Ensure same number of features
        if len(student_features) != len(teacher_features):
            raise ValueError(f"Mismatch in number of features: student has {len(student_features)}, "
                           f"teacher has {len(teacher_features)}")
        
        total_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # TODO: Add dimension matching logic here when needed
            # (stride and channel dimension matching will be added to Teacher class)
            
            if self.loss_type == 'mse':
                loss = F.mse_loss(s_feat, t_feat)
            
            elif self.loss_type == 'cosine':
                # Spatial cosine similarity (preserves spatial structure)
                # Shape: [B, C, H, W]
                B, C, H, W = s_feat.shape
                
                # Reshape to [B, C, H*W] and transpose to [B, H*W, C]
                s_reshaped = s_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
                t_reshaped = t_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
                
                # Normalize features (L2 normalization per spatial location)
                s_norm = F.normalize(s_reshaped, p=2, dim=2)  # [B, H*W, C]
                t_norm = F.normalize(t_reshaped, p=2, dim=2)  # [B, H*W, C]
                
                # Compute cosine similarity at each spatial location
                # Einstein sum: batch, spatial, channels -> batch, spatial
                cosine_sim = torch.einsum('bpc,bpc->bp', s_norm, t_norm)  # [B, H*W]
                
                # Loss is 1 - mean(cosine_similarity)
                # Average over spatial locations and batch
                loss = 1.0 - cosine_sim.mean()
            
            elif self.loss_type == 'cosine_global':
                # Global pooling version (alternative option)
                s_flat = s_feat.flatten(2).mean(dim=2)  # [B, C]
                t_flat = t_feat.flatten(2).mean(dim=2)  # [B, C]
                loss = 1.0 - F.cosine_similarity(s_flat, t_flat, dim=1).mean()
            
            elif self.loss_type == 'kl_div':
                # Apply softmax with temperature
                s_soft = F.log_softmax(s_feat.flatten(2) / self.temperature, dim=1)
                t_soft = F.softmax(t_feat.flatten(2) / self.temperature, dim=1)
                loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.temperature ** 2)
            
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss += loss
        
        return total_loss / len(student_features)
    
    def training_step(self, batch, batch_idx):
        """Training step for knowledge distillation.
        
        Args:
            batch: Input batch containing images and optional labels
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        
        # Get student features
        student_features = self.student(images)
        
        # Get teacher features (no gradient)
        with torch.no_grad():
            teacher_features = self.teacher(images)
        
        # Compute distillation loss
        loss = self.compute_distillation_loss(student_features, teacher_features)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: Input batch containing images and optional labels
            batch_idx: Batch index
        """
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        
        # Get student features
        student_features = self.student(images)
        
        # Get teacher features (no gradient)
        with torch.no_grad():
            teacher_features = self.teacher(images)
        
        # Compute distillation loss
        loss = self.compute_distillation_loss(student_features, teacher_features)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer configuration
        """
        optimizer_cfg = self.config['training']['optimizer']
        optimizer_name = optimizer_cfg['name'].lower()
        lr = optimizer_cfg['lr']
        weight_decay = optimizer_cfg['weight_decay']
        
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = optimizer_cfg.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.student.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
