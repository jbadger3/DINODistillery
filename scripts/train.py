"""
Training script for DINO distillation
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import load_yaml
from lightning_module import DistillationLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description='Train DINO distillation model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def setup_callbacks(config: dict, output_dir: str):
    """Setup training callbacks.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for checkpoints
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Checkpoint callback
    logging_cfg = config['logging']
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='{epoch}-{val_loss:.4f}',
        monitor=logging_cfg['checkpoint_monitor'],
        mode=logging_cfg['checkpoint_mode'],
        save_top_k=logging_cfg['save_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_cfg = config['early_stopping']
    if early_stop_cfg.get('enabled', False):
        early_stop_callback = EarlyStopping(
            monitor=early_stop_cfg['monitor'],
            patience=early_stop_cfg['patience'],
            mode=early_stop_cfg['mode'],
            min_delta=early_stop_cfg.get('min_delta', 0.0),
            verbose=early_stop_cfg.get('verbose', True)
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


def main():
    args = parse_args()
    
    print(f"Starting training with config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    # Load configuration
    config = load_yaml(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create lightning module
    print("Creating distillation module...")
    model = DistillationLightningModule(config)
    
    # Setup logger
    logging_cfg = config['logging']
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=logging_cfg['experiment_name'],
        version=None
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config, args.output_dir)
    
    # Create trainer
    training_cfg = config['training']
    trainer = L.Trainer(
        max_epochs=training_cfg['max_epochs'],
        accelerator=training_cfg['accelerator'],
        devices=training_cfg['devices'],
        precision=training_cfg['precision'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # TODO: Setup dataloaders
    print("\nNote: Dataloader setup is not yet implemented.")
    print("Model creation successful!")
    
    # Train model (commented until dataloaders are implemented)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
