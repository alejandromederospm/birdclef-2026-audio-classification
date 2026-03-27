<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNet--B2-blueviolet)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-BirdCLEF%202026-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

# BirdCLEF 2026 - Audio Classification

Multi-species audio classification system for the **BirdCLEF 2026** Kaggle competition. The model identifies **234 species** (birds, amphibians, mammals, reptiles, and insects) from soundscape recordings captured in the **Pantanal wetlands**, one of the most biodiverse ecosystems on Earth.

## Architecture

```
                         ┌──────────────────────────┐
                         │   Raw Audio (.ogg/.wav)   │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   audiomentations          │
                         │   (Gaussian noise, pitch   │
                         │    shift, time stretch)     │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   Mel Spectrogram          │
                         │   Extraction (librosa)     │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   EfficientNet-B2 (timm)   │
                         │   CNN Backbone              │
                         │   (pretrained ImageNet)     │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   Classification Head       │
                         │   (234 species output)      │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   TTA + Ensemble            │
                         │   (K-fold predictions)      │
                         └──────────────────────────┘
```

## Key Features

- **Mel Spectrogram Extraction** -- converts raw audio into image-like representations for CNN processing
- **EfficientNet-B2 Backbone** -- pretrained CNN via `timm` for transfer learning on spectrograms
- **audiomentations Augmentation** -- domain-specific audio augmentations (noise injection, pitch shift, time stretch)
- **StratifiedGroupKFold** -- ensures balanced species distribution across folds while preventing data leakage by grouping on recording ID
- **BCEWithLabelSmoothing** -- binary cross-entropy with label smoothing for multi-label classification
- **Warmup + CosineAnnealing Scheduler** -- learning rate warm-up followed by cosine annealing for stable training
- **Mixed Precision (AMP)** -- automatic mixed precision training for faster training and reduced memory usage
- **K-Fold Training** -- multiple fold models for robust ensemble predictions
- **Test-Time Augmentation (TTA)** -- augmented inference for improved predictions at test time
- **MLflow Tracking** -- experiment tracking for hyperparameters, metrics, and artifacts

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download the competition data from [Kaggle BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) and place it in the `data/` directory:

```
data/
├── train_audio/
├── test_soundscapes/
├── train_metadata.csv
└── sample_submission.csv
```

### 3. Train

```bash
python train.py --config configs/efficientnet_b2.yaml --fold 0
```

### 4. Inference

```bash
python inference.py --checkpoint checkpoints/best_model_fold0.pt --tta
```

### 5. Track experiments

```bash
mlflow ui --port 5000
```

## Results

| Model | Fold | ROC-AUC (macro) | Notes |
|-------|------|-----------------|-------|
| EfficientNet-B2 | 0 | _TBD_ | Baseline |
| EfficientNet-B2 | Ensemble | _TBD_ | K-fold ensemble + TTA |

> Metric: **macro ROC-AUC** (competition evaluation metric)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | PyTorch 2.x |
| Audio Processing | librosa, torchaudio |
| Augmentation | audiomentations |
| CNN Backbone | EfficientNet-B2 (timm) |
| Experiment Tracking | MLflow |
| Validation | StratifiedGroupKFold (scikit-learn) |
| Visualization | matplotlib, seaborn |

## Project Structure

```
birdclef-2026-audio-classification/
├── configs/              # Training configuration files
├── data/                 # Competition data (not tracked)
├── notebooks/            # EDA and experiment notebooks
├── src/                  # Source code
│   ├── dataset.py        # Dataset and data loaders
│   ├── model.py          # Model architecture
│   ├── transforms.py     # Audio augmentations
│   ├── loss.py           # Loss functions
│   └── utils.py          # Utility functions
├── train.py              # Training script
├── inference.py          # Inference script
├── requirements.txt
└── README.md
```

---

# BirdCLEF 2026 - Clasificacion de Audio (ES)

Sistema de clasificacion de audio multi-especie para la competencia **BirdCLEF 2026** de Kaggle. El modelo identifica **234 especies** (aves, anfibios, mamiferos, reptiles e insectos) a partir de grabaciones de paisajes sonoros capturados en los **humedales del Pantanal**, uno de los ecosistemas mas biodiversos del planeta.

## Descripcion

Este proyecto utiliza tecnicas de aprendizaje profundo para clasificar especies a partir de grabaciones de audio. El pipeline convierte audio crudo en espectrogramas Mel, que luego se procesan mediante una red neuronal convolucional EfficientNet-B2 preentrenada. Se emplean multiples tecnicas de augmentacion, validacion cruzada estratificada y ensamblaje de modelos para maximizar el rendimiento.

## Caracteristicas Principales

- **Extraccion de Espectrogramas Mel** -- convierte audio en representaciones tipo imagen para procesamiento CNN
- **Backbone EfficientNet-B2** -- CNN preentrenada via `timm` para transfer learning
- **Augmentacion con audiomentations** -- augmentaciones especificas de audio (inyeccion de ruido, cambio de tono, estiramiento temporal)
- **StratifiedGroupKFold** -- distribucion balanceada de especies entre folds sin fuga de datos
- **Entrenamiento con Precision Mixta (AMP)** -- entrenamiento mas rapido y menor uso de memoria
- **TTA (Test-Time Augmentation)** -- augmentacion en inferencia para predicciones mas robustas
- **Seguimiento con MLflow** -- tracking de experimentos, hiperparametros y metricas

## Como Ejecutar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Preparar datos

Descargar los datos de la competencia desde [Kaggle BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) y colocarlos en el directorio `data/`.

### 3. Entrenar

```bash
python train.py --config configs/efficientnet_b2.yaml --fold 0
```

### 4. Inferencia

```bash
python inference.py --checkpoint checkpoints/best_model_fold0.pt --tta
```

## Resultados

| Modelo | Fold | ROC-AUC (macro) | Notas |
|--------|------|-----------------|-------|
| EfficientNet-B2 | 0 | _POR DEFINIR_ | Linea base |
| EfficientNet-B2 | Ensemble | _POR DEFINIR_ | Ensamble K-fold + TTA |

> Metrica: **macro ROC-AUC** (metrica de evaluacion de la competencia)
