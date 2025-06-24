"""
Módulo de configuración Hydra para los experimentos de mariposas.
"""

import os
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import warnings
warnings.filterwarnings('ignore')


def setup_hydra_config(config_path: str = "config", config_name: str = "config") -> DictConfig:
    """Inicializa y carga la configuración de Hydra."""
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
    
    return cfg


def setup_experiment_config(experiment_name: str) -> DictConfig:
    """Carga la configuración específica de un experimento."""
    config_path = f"config/{experiment_name}"
    return setup_hydra_config(config_path=config_path, config_name="config")


def configure_device_and_training(cfg: DictConfig) -> dict:
    """Configura el dispositivo y parámetros de entrenamiento."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'cuda_version': torch.version.cuda,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'mixed_precision_available': hasattr(torch.cuda, "amp")
        }
        
        if gpu_info['memory_gb'] >= 8:
            batch_size = 64
        elif gpu_info['memory_gb'] >= 4:
            batch_size = 32
        else:
            batch_size = 16
    else:
        batch_size = 16
    
    if torch.cuda.is_available() and cfg.device.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    pl.seed_everything(cfg.seed)
    
    return {
        'device': device,
        'batch_size': batch_size,
        'gpu_info': gpu_info,
        'precision': cfg.device.precision if torch.cuda.is_available() else '32',
        'accelerator': 'auto',
        'devices': 'auto'
    }


def setup_wandb_logging(cfg: DictConfig, experiment_config: dict = None) -> dict:
    """Configura Weights & Biases logging."""
    wandb_config = {
        'project': cfg.logging.project_name,
        'name': cfg.logging.experiment_name,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    if experiment_config:
        wandb_config['config'].update(experiment_config)
    
    if hasattr(cfg.logging, 'tags'):
        wandb_config['tags'] = cfg.logging.tags
        
    return wandb_config


def get_data_module_config(cfg: DictConfig, device_config: dict) -> dict:
    """Extrae la configuración para el DataModule."""
    return {
        'data_dir': cfg.data.data_dir,
        'metadata_csv': cfg.data.metadata_csv,
        'batch_size': device_config['batch_size'],
        'num_workers': cfg.data.num_workers,
        'image_size': cfg.data.image_size,
        'labeled_ratio': cfg.data.splits.labeled_ratio,
        'seed': cfg.seed
    }


def get_model_config(cfg: DictConfig) -> dict:
    """Extrae la configuración del modelo."""
    model_config = {
        'n_channels': cfg.model.architecture.n_channels,
        'learning_rate': cfg.training.learning_rate
    }
    
    if cfg.model.name == "DenoisingUNetAutoencoder":
        model_config.update({
            'noise_prob': cfg.model.noise.probability
        })
    elif cfg.model.name == "ButterflyClassifier":
        model_config.update({
            'num_classes': cfg.model.classifier.num_classes,
            'dropout': cfg.model.classifier.dropout
        })
    
    return model_config


def get_trainer_config(cfg: DictConfig, device_config: dict) -> dict:
    """Extrae la configuración del PyTorch Lightning Trainer."""
    return {
        'max_epochs': cfg.training.max_epochs,
        'accelerator': device_config['accelerator'],
        'devices': device_config['devices'],
        'precision': device_config['precision'],
        'gradient_clip_val': cfg.training.lightning.gradient_clip_val,
        'accumulate_grad_batches': cfg.training.lightning.accumulate_grad_batches,
        'check_val_every_n_epoch': cfg.training.validation.check_val_every_n_epoch,
        'val_check_interval': cfg.training.validation.val_check_interval
    }


def print_config_summary(cfg: DictConfig, device_config: dict):
    """Imprime un resumen de la configuración cargada."""
    print("=" * 60)
    print("CONFIGURACIÓN CARGADA CON HYDRA")
    print("=" * 60)
    
    if hasattr(cfg, 'experiment'):
        print(f"Experimento: {cfg.experiment.name}")
        print(f"Descripción: {cfg.experiment.description}")
    
    print(f"\nDispositivo: {device_config['device']}")
    if device_config['gpu_info']:
        print(f"GPU: {device_config['gpu_info']['name']}")
        print(f"Memoria GPU: {device_config['gpu_info']['memory_gb']:.1f} GB")
        print(f"CUDA: {device_config['gpu_info']['cuda_version']}")
    
    print(f"\nModelo: {cfg.model.name}")
    print(f"Batch Size: {device_config['batch_size']}")
    print(f"Precisión: {device_config['precision']}")
    
    print(f"\nEpocas: {cfg.training.max_epochs}")
    print(f"Learning Rate: {cfg.training.learning_rate}")
    print(f"Optimizador: {cfg.training.optimizer.name}")
    
    print(f"\nTamaño de imagen: {cfg.data.image_size}")
    print(f"Ratio etiquetado: {cfg.data.splits.labeled_ratio}")
    print(f"Workers: {cfg.data.num_workers}")
    
    print("=" * 60)


def override_config(cfg: DictConfig, overrides: dict) -> DictConfig:
    """Permite sobrescribir valores de configuración dinámicamente."""
    cfg_copy = OmegaConf.create(cfg)
    
    for key, value in overrides.items():
        OmegaConf.set_struct(cfg_copy, False)
        keys = key.split('.')
        current = cfg_copy
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        OmegaConf.set_struct(cfg_copy, True)
    
    return cfg_copy


def setup_experiment_1_config(data_split: str = "70_30") -> tuple:
    """Configura el Experimento 1 con Hydra."""
    cfg = setup_experiment_config("experiment_1")
    
    if data_split != "70_30":
        split_config = cfg.data_splits[f"split_{data_split}"]
        cfg = override_config(cfg, {
            'data.splits.labeled_ratio': split_config.labeled_ratio
        })
    
    device_config = configure_device_and_training(cfg)
    wandb_config = setup_wandb_logging(cfg, {'data_split': data_split})
    
    return cfg, device_config, wandb_config


def setup_experiment_2_config() -> tuple:
    """Configura el Experimento 2 con Hydra."""
    cfg = setup_experiment_config("experiment_2")
    device_config = configure_device_and_training(cfg)
    wandb_config = setup_wandb_logging(cfg)
    
    return cfg, device_config, wandb_config 