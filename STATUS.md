# SCL_2.0 - Estado del Proyecto

## Resumen de lo implementado

### 1. Representación Simbólica de Acordes
- **`chords/vocab.py`**: Vocabulario de 12 roots, 8 qualities, 7 seventh types, extensiones, alteraciones, added tones, bajo
- **`chords/chord_encoder.py`**: Parseo, encoding (48-dim one-hot/multi-hot) y decoding de chord strings
- Dimensión total: 48 (12 root + 8 quality + 5 seventh + 3 extensions + 4 alterations + 3 added + 13 bass)

### 2. Dataset
- **`Dataset/dataset.parquet`**: 877,342 progresiones originales (Chordonomicon v2)
- **`Dataset/dataset_symbolic.parquet`**: Mismos datos con encoding simbólico 48-dim
- **`Dataset/dataset_with_jsymbolic.parquet`**: 11 features jSymbolic extraídas
- **`Dataset/dataset_conditioned.parquet`**: Features PCS + 4 perceptuales normalizadas (5-dim conditioning)
- **`Dataset/dataset_conditioned_100k.parquet`**: Subset de 100K para entrenamiento rápido
- **`cvae/dataset.py`**: `ChordProgressionDataset` con soporte opcional de conditioning

### 3. Modelo CVAE
- **`cvae/model.py`**: 
  - `ChordEncoder`: LSTM bidireccional (2 layers, 256 hidden) → mu/logvar (64-dim latent)
  - `ChordDecoder`: LSTM condicionado (2 layers, 256 hidden) → logits 48-dim
  - `CVAE`: Encoder + Decoder con reparameterization y concatenación de conditioning
  - 3,186,864 parámetros totales

### 4. Loss Functions (implementadas en `cvae/losses.py`)
| Loss | Símbolo | Propósito | Peso |
|------|---------|-----------|------|
| Reconstruction | L_recon | BCE entre logits y target (padding ignorado) | 1.0 |
| KL Divergence | L_KL | Regularización hacia N(0,I) con free-bits threshold | β=0.1 |
| Coherence | L_coh | Penaliza acordes fuera de tonalidad (Krumhansl-Schmuckler) | λ₁=0.1 |
| Tension | L_tens | Penaliza acordes sin 7mas/extensiones (< 4 pitch classes) | λ₂=0.05 |
| Movement | L_mov | Penaliza saltos de root > 7 semitonos | λ₃=0.1 |

### 5. Perceptual Complexity Score (PCS)
- **`PCS/pcs.py`**: `PerceptualComplexityScore` con pesos derivados de encuesta perceptual
- **`PCS/pcs_weights.json`**: 4 features activas:
  - `Seventh_Chords` (7C): peso 0.443
  - `Variability_of_Number_of_Simultaneous_Pitch_Classes` (VNSPC): peso 0.378
  - `Distance_Between_Two_Most_Common_Vertical_Intervals` (DTMCVI): peso 0.338
  - `Vertical_Dissonance_Ratio` (VDR): peso 0.201
- **`cvae/prepare_conditioning.py`**: Script que calcula PCS y normaliza features perceptuales

### 6. Entrenamiento (GPU: RTX 4070 8GB)
- **`scripts/train.py`**: Script completo con:
  - Soporte conditioning (--condition-dim 5 para PCS)
  - Checkpoint resume (--resume)
  - TensorBoard logging
  - Early stopping con patience
  - Gradiente clipping
- Checkpoints guardados en `checkpoints/cvae_pcs_100k/`

#### Curva de entrenamiento (100K subset):
```
Epoch  | Train Total | Recon    | KL     | Coh    | Val Total
-------|-------------|----------|--------|--------|----------
     1 |      5.2065 |   5.1683 | 0.2322 | 0.1495 |    3.1418
     2 |      1.6203 |   1.5848 | 0.0005 | 0.3543 |    0.4931
     3 |      0.3575 |   0.3149 | 0.0000 | 0.4255 |    0.1813
     4 |      0.1774 |   0.1346 | 0.0000 | 0.4280 |    0.0745
     5 |      0.1032 |   0.0604 | 0.0000 | 0.4283 |    0.0343
     6 |      0.0738 |   0.0310 | 0.0000 | 0.4286 |    0.0177
     7 |      0.0604 |   0.0176 | 0.0000 | 0.4286 |    0.0101
     8 |      0.0536 |   0.0107 | 0.0000 | 0.4288 |    0.0061
     9 |      0.0498 |   0.0069 | 0.0000 | 0.4286 |    0.0039
    10 |      0.0474 |   0.0045 | 0.0000 | 0.4288 |    0.0023
    11 |      0.0459 |   0.0030 | 0.0000 | 0.4287 |    0.0016
    12 |      0.0450 |   0.0021 | 0.0000 | 0.4287 |    0.0010
    13 |      0.0443 |   0.0014 | 0.0000 | 0.4287 |    0.0007
    14 |      0.0439 |   0.0010 | 0.0000 | 0.4286 |    0.0004
    15 |      0.0436 |   0.0007 | 0.0000 | 0.4287 |    0.0003
    16 |      0.0434 |   0.0005 | 0.0000 | 0.4288 |    0.0002
```

### 7. Generación
- **`scripts/generate.py`**: Generación de progresiones desde checkpoint
- **`scripts/reharmonize.py`**: Reharmonización condicionada por PCS

## Problema Identificado: Posterior Collapse

**Síntoma**: KL≈0 desde época 2 → modelo ignora latent z y conditioning PCS.

**Causa Raíz**: El decoder recibe teacher forcing completo (acorde real en cada step). Con 256-dim LSTM, aprende mapeo directo acorde→acorde sin usar z ni conditioning.

**Evidencia**:
- Reconstrucción perfecta (recon=0.0005)
- PCS no afecta output (cambiarlo de 0.0 a 0.8 no altera resultado)
- Generación con z∼N(0,I) produce solo acordes C

## Próximos Pasos (Pendientes)

1. **Arreglar posterior collapse**:
   - Opción A: KL annealing (β=0→1 gradual) + reducir teacher forcing a 0.2 + aumentar free-bits a 1.0
   - Opción B: Rediseñar decoder sin teacher forcing (solo SOS + z_cond)
   - Opción C: β=1.0, teacher_forcing=0.1, reentrenar desde cero

2. **Mejorar coherence loss** (subió de 0.15→0.43 en vez de bajar):
   - Investigar por qué empeora con el entrenamiento
   - Posible ajuste de λ_coh

3. **Entrenar en dataset completo** (877K progresiones):
   - GPU: 8 épocas en 2h para 100K → estimado ~17h para full dataset
   - Usar resume desde checkpoint actual

## Archivos Clave

```
SCL_2.0/
├── chords/
│   ├── vocab.py              # Vocabulario y dimensiones
│   └── chord_encoder.py       # Parseo, encode/decode de acordes
├── cvae/
│   ├── config.py              # Hiperparámetros
│   ├── model.py               # CVAE (Encoder + Decoder)
│   ├── dataset.py             # Dataset + DataLoader con conditioning
│   ├── losses.py              # Losses (recon, KL, coherence, tension, movement)
│   └── prepare_conditioning.py # Preparación dataset condicionado
├── PCS/
│   ├── pcs.py                 # PerceptualComplexityScore
│   └── pcs_weights.json       # Pesos del PCS
├── scripts/
│   ├── train.py               # Entrenamiento con resume y conditioning
│   ├── generate.py            # Generación desde checkpoint
│   └── reharmonize.py         # Reharmonización condicionada
├── Dataset/
│   ├── dataset_conditioned.parquet       # 877K con PCS + features
│   └── dataset_conditioned_100k.parquet  # Subset 100K
├── checkpoints/
│   └── cvae_pcs_100k/
│       ├── best.pt   # Epoch 16 (val_loss=0.0002)
│       └── last.pt   # Epoch 16
└── runs/
    └── cvae_pcs_100k/  # TensorBoard logs
```
