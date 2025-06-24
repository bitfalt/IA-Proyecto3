# Configuración Hydra - Proyecto Mariposas

## Estructura del Proyecto

```
config/
├── config.yaml                 # Configuración principal
├── base_config.yaml           # Configuración base compartida
├── data/
│   └── butterfly_dataset.yaml # Configuración de datos
├── model/
│   ├── unet_autoencoder.yaml  # U-Net Autoencoder
│   └── denoising_autoencoder.yaml # Denoising Autoencoder
├── training/
│   └── default.yaml           # Configuración de entrenamiento
├── experiment_1/
│   └── config.yaml           # Experimento 1 específico
└── experiment_2/
    └── config.yaml           # Experimento 2 específico
```

## Módulo Principal

`hydra_config.py` contiene las funciones principales:

- `setup_experiment_1_config(data_split)` - Configura Experimento 1
- `setup_experiment_2_config()` - Configura Experimento 2
- `configure_device_and_training(cfg)` - Configuración automática GPU/CPU
- `setup_wandb_logging(cfg)` - Configuración Wandb
- `override_config(cfg, overrides)` - Modificaciones dinámicas

## Uso en Notebooks

### Experimento 1 (Transfer Learning)

```python
from hydra_config import setup_experiment_1_config, print_config_summary

# Cargar configuración
cfg, device_config, wandb_config = setup_experiment_1_config(data_split="70_30")

# Mostrar resumen
print_config_summary(cfg, device_config)

# Usar configuraciones
data_module_config = get_data_module_config(cfg, device_config)
model_config = get_model_config(cfg)
trainer_config = get_trainer_config(cfg, device_config)
```

### Experimento 2 (Denoising + Clustering)

```python
from hydra_config import setup_experiment_2_config, print_config_summary

# Cargar configuración
cfg, device_config, wandb_config = setup_experiment_2_config()

# Mostrar resumen
print_config_summary(cfg, device_config)

# Usar configuraciones
data_module_config = get_data_module_config(cfg, device_config)
model_config = get_model_config(cfg)
trainer_config = get_trainer_config(cfg, device_config)
```

## Configuraciones Disponibles

### Splits de Datos (Experimento 1)
- `70_30`: 30% etiquetado, 70% no etiquetado
- `80_20`: 20% etiquetado, 80% no etiquetado  
- `90_10`: 10% etiquetado, 90% no etiquetado

### Clasificadores (Experimento 1)
- **Clasificador A**: Sin preentrenamiento
- **Clasificador B1**: Encoder congelado
- **Clasificador B2**: Fine-tuning completo

### Configuración de Ruido (Experimento 2)
- Tipo: Salt & Pepper
- Probabilidad: 0.05

### Configuración t-SNE
- Perplexity: 30
- Iteraciones: 1000
- Learning Rate: 200

### Configuración K-means
- Clusters: 30
- Random State: 42

## Overrides Dinámicos

```python
from hydra_config import override_config

# Modificar configuración
new_cfg = override_config(cfg, {
    'training.max_epochs': 100,
    'training.learning_rate': 1e-4,
    'data.batch_size': 64
})
```

## Dependencias

```bash
pip install -r requirements_hydra.txt
```

Incluye:
- hydra-core
- omegaconf
- pytorch-lightning
- wandb 