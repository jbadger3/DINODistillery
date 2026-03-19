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
from dataloader import create_sa1b_dataloaders_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train DINO distillation model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def setup_callbacks(config: dict, experiment_dir: str):
    """Setup training callbacks.
    
    Args:
        config: Configuration dictionary
        experiment_dir: Experiment directory (outputs/experiment_name/)
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Checkpoint callback
    logging_cfg = config['logging']
    eval_cfg = config.get('evaluation', {})
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_monitor = logging_cfg['checkpoint_monitor']
    checkpoint_mode = logging_cfg['checkpoint_mode']
    monitored_save_on_train_epoch_end = logging_cfg.get('save_on_train_epoch_end', False)
    configured_every_n_epochs = int(logging_cfg.get('every_n_epochs', 1))
    eval_every_n_epochs = int(eval_cfg.get('eval_every_n_epochs', 1))

    # Keep checkpoint cadence aligned with validation cadence for val_* monitors.
    if checkpoint_monitor.startswith('val_'):
        monitored_every_n_epochs = eval_every_n_epochs
        if configured_every_n_epochs != eval_every_n_epochs:
            print(
                "INFO: checkpoint_monitor is validation-based; using "
                "evaluation.eval_every_n_epochs for monitored checkpoint cadence "
                f"({eval_every_n_epochs}) instead of logging.every_n_epochs ({configured_every_n_epochs})."
            )
    else:
        monitored_every_n_epochs = configured_every_n_epochs

    # Avoid missing-metric warnings when monitoring validation metrics from train epoch end
    if checkpoint_monitor.startswith('val_') and monitored_save_on_train_epoch_end:
        print(
            "WARNING: logging.save_on_train_epoch_end=true with checkpoint_monitor='val_*' "
            "can trigger missing metric warnings before validation runs. "
            "Overriding monitored checkpoint callback to save_on_train_epoch_end=false."
        )
        monitored_save_on_train_epoch_end = False

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:03d}-{val_loss:.4f}',
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=logging_cfg['save_top_k'],
        every_n_epochs=monitored_every_n_epochs,
        save_on_train_epoch_end=monitored_save_on_train_epoch_end,
        save_last=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Optional recovery checkpoint callback (train-epoch based, no monitor)
    # Purpose: preserve resumable checkpoints even if validation has not run yet.
    # Uses Lightning's built-in rolling last.ckpt behavior.
    recovery_cfg = logging_cfg.get('recovery_checkpoint', {})
    if recovery_cfg.get('enabled', True):
        recovery_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='recovery-epoch{epoch:03d}',
            monitor=None,
            mode='min',
            save_top_k=0,
            every_n_epochs=recovery_cfg.get('every_n_epochs', 1),
            save_on_train_epoch_end=True,
            save_last=True,
            verbose=True
        )
        callbacks.append(recovery_callback)
    
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
    
    # Load configuration
    config = load_yaml(args.config)
    
    # Build experiment directory structure (hardcoded outputs/ base)
    logging_cfg = config['logging']
    experiment_name = logging_cfg['experiment_name']
    experiment_dir = os.path.join('outputs', experiment_name)
    
    print(f"Starting training with config: {args.config}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"  - Checkpoints: {os.path.join(experiment_dir, 'checkpoints')}")
    print(f"  - Logs: {os.path.join(experiment_dir, 'logs')}")
    
    # Create experiment directory structure
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    
    # Create lightning module
    print("\nCreating distillation module...")
    model = DistillationLightningModule(config)
    
    # Setup logger (logs go in experiment_dir/logs/)
    logger = TensorBoardLogger(
        save_dir=experiment_dir,
        name='logs',
        version=None
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config, experiment_dir)
    
    # Create trainer
    training_cfg = config['training']
    eval_cfg = config.get('evaluation', {})
    trainer = L.Trainer(
        max_epochs=training_cfg['max_epochs'],
        accumulate_grad_batches=training_cfg.get('accumulate_grad_batches', 1),
        accelerator=training_cfg['accelerator'],
        devices=training_cfg['devices'],
        precision=training_cfg['precision'],
        check_val_every_n_epoch=eval_cfg.get('eval_every_n_epochs', 1),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Setup dataloaders
    print("\nSetting up dataloaders...")
    train_loader, val_loader = create_sa1b_dataloaders_from_config(config)
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print(f"Total train images: {len(train_loader.dataset):,}")
    print(f"Total val images: {len(val_loader.dataset):,}")
    
    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=args.resume
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best checkpoint saved in: {os.path.join(experiment_dir, 'checkpoints')}")
    print(f"TensorBoard logs in: {os.path.join(experiment_dir, 'logs')}")
    print("="*60)


if __name__ == "__main__":
    main()
