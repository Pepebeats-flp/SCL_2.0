# SCL 2.0 — Symbolic Chord Language

VAE condicionado para progresiones de acordes simbólicas, con control de complejidad perceptual y enrichment por gradient ascent en espacio latente.

---

## Índice

1. [Pipeline de Datos](#1-pipeline-de-datos)
2. [Codificación de Acordes](#2-codificación-de-acordes)
3. [Perceptual Complexity Score (PCS)](#3-perceptual-complexity-score-pcs)
4. [Arquitectura del Modelo](#4-arquitectura-del-modelo)
5. [Entrenamiento](#5-entrenamiento)
6. [Logs y Métricas](#6-logs-y-métricas)
7. [Enrichment Pipeline](#7-enrichment-pipeline)
8. [Pipeline de Inferencia Completa](#8-pipeline-de-inferencia-completa)
9. [PCS: Dos Variantes (v1 y v2)](#9-pcs-dos-variantes-v1-y-v2)
10. [Experimentos y Evaluación](#10-experimentos-y-evaluación)
11. [Scripts / Cheatsheet](#11-scripts--cheatsheet)

---

## 1. Pipeline de Datos

```
MIDI (.mid) → jSymbolic → dataset_with_jsymbolic.parquet
                                      ↓
MIDI (.mid) → music21   → dataset_symbolic.parquet
                                      ↓
                           dataset_conditioned.parquet
```

### Etapa 1 — Extracción con jSymbolic

```
Dataset/generate_dataset_with_jsymbolic.py
```

Usa **jSymbolic_2_2_user/jSymbolic2.jar** (Java). Procesa ~877k MIDIs del dataset Lakh y extrae **1496 features** por progresión: intervalos verticales, densidad de pitch classes, disonancia, etc.

Output: `Dataset/dataset_with_jsymbolic.parquet` (877k filas × 1496 columnas + `id`).

### Etapa 2 — Codificación simbólica

Se parsean los MIDIs con `music21`, extrayendo acorde por acorde como vectores de 48 bits. Cada progresión se guarda como un JSON de arrays `[[48-dim], [48-dim], ...]`.

Output: `Dataset/dataset_symbolic.parquet` con columnas:
- `id` — hash único del MIDI
- `symbolic` — JSON de la secuencia de vectores 48-dim
- `n_chords` — cantidad de acordes
- `chords` — nombres legibles tipo `"C | Am | F | G7"`

### Etapa 3 — Conditioning (PCS)

```
cvae/prepare_conditioning.py
```

Junta ambos datasets por `id`. Calcula el PCS (Perceptual Complexity Score) y sus 4 dimensiones a partir de features de jSymbolic. Normaliza cada dimensión a [0, 1] con min-max.

Output: `Dataset/dataset_conditioned.parquet` — filtra solo los 877k con features, agrega:
- `pcs` — score compuesto (0–1)
- `7C` — Seventh Chords
- `VNSPC` — Variability of # Simultaneous Pitch Classes
- `DTMCVI` — Distance Between Two Most Common Vertical Intervals
- `VDR` — Vertical Dissonance Ratio

---

## 2. Codificación de Acordes

Cada acorde se representa como un vector binario de **48 dimensiones**, dividido en slots:

| Slot | Dims | Contenido | Ejemplo |
|------|------|-----------|---------|
| `root` | 0–11 | Nota fundamental (C–B one-hot) | C → `[1,0,...]` |
| `quality` | 12–19 | Tipo: maj, min, dim, aug, sus2, sus4, no3d, other | min → `[0,1,0,...]` |
| `seventh` | 20–24 | Séptima: none, dom7, maj7, dim7, aug7 | dom7 → `[0,1,0,0,0]` |
| `ext` | 25–27 | Extensiones: 9, 11, 13 (multilabel) | `[1,0,0]` = 9 |
| `alt` | 28–31 | Alteraciones: b9, #9, #11, b13 (multilabel) | |
| `added` | 32–34 | Added: add9, add11, add13 (multilabel) | |
| `bass` | 35–47 | Bajo (C–B + "no bass") | C → `[1,0,...]` |

Total: 12 + 8 + 5 + 3 + 4 + 3 + 13 = **48 bits**.

### Parseo y encoding

- **`parse_chord("Cmaj7")`** → diccionario con root, quality, seventh, extensions, alterations, added, bass
- **`encode_chord(parsed)`** → vector 48-dim
- **`decode_chord(vec)`** → string tipo `"Cmaj7"` o `"G7/B"`
- **`progression_to_encoding(["C","Am","F","G7"])`** → array numpy `(4, 48)`

---

## 3. Perceptual Complexity Score (PCS)

Derivado de un estudio perceptual: humanos calificaron progresiones en baja/media/alta complejidad. Se correlacionaron las calificaciones con ~10 features de jSymbolic y se quedaron las 4 dimensiones con correlaciones significativas (p < 0.05).

### 4 dimensiones perceptuales

| Dim | Nombre | Peso | Descripción |
|-----|--------|------|-------------|
| **7C** | Seventh Chords | 0.443 | Proporción de acordes con séptima |
| **VNSPC** | Variability of # Simultaneous Pitch Classes | 0.378 | Qué tanto varía la densidad armónica |
| **DTMCVI** | Distance Between Two Most Common Vertical Intervals | 0.338 | Diversidad de intervalos armónicos |
| **VDR** | Vertical Dissonance Ratio | 0.201 | Balance disonancia/consonancia |

### Cálculo (`PCS/pcs.py`)

1. Peso de cada feature derivado de la correlación media con ratings humanos
2. Si la correlación es negativa, se invierte la feature
3. Cada feature se normaliza a [0, 1]
4. Score = suma ponderada / suma de pesos
5. Clip a [0, 1]

---

## 4. Arquitectura del Modelo

### RVAE — Conditioned LSTM VAE

```
Encoder: LSTM bidireccional (hidden=256, 2 capas) → mu(128), logvar(128)
                    ↑
            x = secuencia de acordes (T×48)

Prior: MLP(c_perceptual[4]) → 16 → ReLU → 128 → ReLU → mu_prior(128), logvar_prior(128)

Latent: z = mu + ε · exp(0.5·logvar)    ε ~ N(0, I)
        z_dec = z    (z_only_decoder=True, la info de C fluye por z)

Decoder: LSTM (hidden=256, 2 capas) con z_dec expandido como condicion → logits(48)
         ↑ teacher forcing con scheduled sampling (1.0 → 0.3 en 15 épocas)
         ↑ word dropout (70% de inputs se tiran a cero en training)

c_predictor: MLP(z) → 64 → ReLU → 64 → ReLU → c_pred[4]
             predice las 4 dims perceptuales desde z (entrenado end-to-end)

key_predictor: MLP(z) → 64 → ReLU → 64 → ReLU → key_logits[12]
               predice la tonalidad desde z

ComplexityPrior: aprende µ_prior(c) y σ_prior(c) para cada c condición — permite samplear
                 progresiones con nivel de complejidad específico sin tener que codificar
                 una real.
```

### Forward completo

```python
mu, logvar = encoder(x, lengths)
z = reparameterize(mu, logvar)
mu_prior, logvar_prior = prior(c)
c_pred = c_predictor(z)          # reconstruye las 4 dims
key_logits = key_predictor(z)    # reconstruye tonalidad
z_dec = z                        # (z_only_decoder=True)
logits = decoder(z_dec, x, lengths, teacher_forcing_prob)
```

### Decodificador (generate)

```python
z_expandido a cada timestep + LSTM autoregresivo
SOS = vector de cero
for t in range(max_len):
    decoder_in = concat(output_anterior, z)
    h = LSTM(decoder_in, h)
    logits = fc(h)
    probs = sigmoid(logits)
    out = deterministic_decode(probs)  # multinomial root/qual/seventh/bass, threshold ext/alt/added
```

### CPredictor

Es el componente clave del enrichment: un MLP que mapea `z → (7C, VNSPC, DTMCVI, VDR)`. Se entrena junto al VAE con MSE loss sobre las 4 dimensiones reales del dataset. Así el gradiente de `c_predictor` nos dice en qué dirección mover `z` para aumentar una dimensión específica.

### KeyPredictor

MLP auxiliar `z → key_logits[12]` (cross-entropy con la key detectada de la progresión real). Ayuda a que `z` codifique información tonal.

---

## 5. Entrenamiento

### Configuración

| Parámetro | Valor |
|-----------|-------|
| Latent dim | 128 (config) / 32 (checkpoint v13) |
| Hidden dim | 256 |
| Capas LSTM | 2 |
| Dropout | 0.2 |
| Word dropout | 0.7 (70% inputs → cero en training) |
| β target | 0.01 |
| Free bits | 1.0 (total) + 0.25 (per-dim) |
| KL cycle | 10 épocas |
| Scheduled sampling | 1.0 → 0.3 en 15 épocas |
| Optimizer | Adam, lr=5e-4, grad clip=5.0 |
| Batch | 256 |

### Función de pérdida

```
L = recon + β · KL + λ_c_pred · MSE(c_pred, c_true) + λ_key_pred · CE(key_pred, key)
```

- **Recon**: BCE con máscara de longitud (solo tokens válidos)
- **KL**: D_KL(N(µ_post, σ_post) || N(µ_prior, σ_prior)) con free bits
- **c_pred**: MSE entre c_pred(z) y las 4 dims reales
- **key_pred**: Cross-entropy entre key_pred(z) y tonalidad real

### KL Cycle

La β arranca en 0 y sube linealmente hasta `β_target` durante los primeros `kl_cycle` épocas (cíclico: reinicia cada ciclo). Esto evita KL collapse: el decoder aprende a usar z antes de que el KL loss sea fuerte.

### Scheduled Sampling

Ratio de teacher forcing decrece linealmente de 1.0 a 0.3 en las primeras 15 épocas. Al final el decoder genera autoregresivamente el 70% del tiempo, forzándolo a aprender a recuperarse de sus propios errores.

### Quality Guards (early stopping automático)

El trainer detecta y frena entrenamientos fallidos:

| Guard | Condición | Causa probable |
|-------|-----------|----------------|
| KL collapse | KL < 0.001 por 2 épocas | Word dropout muy bajo, β muy baja |
| Cheating | KL < 0.01 y recon < 0.1 | Modelo ignora z, copia input |
| Active units | < 15% de dims con varianza > 0.01 | Latent subutilizado |
| KL real | KL_real < 5.0 | Scheduled sampling muy rápido |
| Coherence worsen | coherent loss sube 3 épocas seguidas | λ_coh mal sintonizado |

### Salida por época

```
Epoch  30 | Train: 4.3073 (recon=3.9676, kl=5.7658) | Val: 4.2376 (recon=3.9353, kl=6.0475) |
β=0.050 tf=0.00 | active=1.00 kl_real=7.371 prior_std=0.694 post_std=0.512 logvar=-1.4
```

### Evolución del entrenamiento (checkpoint v13)

| Época | Train Loss | Val Loss | Recon | KL | β | TF | kl_real |
|-------|-----------|---------|-------|-----|-----|-----|---------|
| 1 | 7.95 | 5.64 | 6.68 | 784 | 0.000 | 0.50 | 672 |
| 5 | 4.94 | 4.52 | 4.45 | 6.76 | 0.025 | 0.28 | 8.40 |
| 10 | 4.58 | 4.45 | 4.22 | 4.81 | 0.050 | 0.00 | 5.97 |
| 20 | 4.41 | 4.34 | 4.10 | 4.88 | 0.050 | 0.00 | 5.99 |
| **30** | **4.31** | **4.24** | **3.97** | **5.77** | **0.050** | **0.00** | **7.37** |

El KL real se mantiene saludable (~6-7), las 128 dims están activas (active=1.00), y el prior se contrae suavemente (prior_std 1.0 → 0.69).

### TensorBoard

Los logs se guardan en `runs/rvae/` con scalars:
- `loss/total`, `loss/recon`, `loss/kl` (train + val)
- `loss/coh`, `loss/tens`, `loss/mov` (si están activos)
- `train/tf_prob`, `train/beta`
- `latent/active_units`, `latent/kl_real`, `latent/mu_std`, `latent/prior_std`, `latent/post_std`

---

## 6. Logs y Métricas

### Métricas de entrenamiento

| Métrica | Qué mide | Rango sano |
|---------|----------|------------|
| `recon` | BCE promedio por token válido | 3.5–5.0 |
| `kl` | KL loss con free bits aplicado | 4–8 |
| `kl_real` | KL total (sin free bits) por sample | 5–50 |
| `active_units` | Fracción de dims latentes con varianza > 0.01 | > 0.70 |
| `prior_std` | Desviación del prior σ | 0.5–1.0 |
| `post_std` | Desviación del posterior σ | 0.5–1.0 |
| `logvar` | Media de logvar (negativo = posterior más angosto que prior) | -2 a 0 |

### Decoder Bypass Test (evaluate script)

Mide si el decoder realmente usa z:

```
recon(z_real) @ tf=0: 4.12 BCE
recon(z=0) @ tf=0:    6.89 BCE
Δ (zero - real):       2.77
```

Si Δ > 0.15 → decoder sí usa z.
Si Δ ≈ 0 → decoder ignora z (KL collapse o cheating).

### Análisis post-entrenamiento

`scripts/evaluate_rvae_v11_mem_safe.py` computa:
- Decoder bypass test
- Latent space stats (PCA, t-SNE)
- Correlación z vs PCS
- Calidad de reconstrucción
- Análisis de enriched progressions

---

## 7. Enrichment Pipeline

### El problema

Tenemos un VAE entrenado con progresiones existentes. Queremos **agregar séptimas** a progresiones de triadas **sin reentrenar**. No podemos simplemente cambiar el conditioning porque el modelo aprendió la distribución conjunta de (z, c, x).

### La solución: gradient ascent sobre z con c_predictor

El `c_predictor` es un MLP que mapea `z → (7C, VNSPC, DTMCVI, VDR)`. Para aumentar 7C:

```
z_opt ← z_orig.clone().requires_grad_(True)
for step in range(90):
    loss = MSE(c_predictor(z_opt)[7C], target) + 0.005 · MSE(z_opt, z_orig)
    loss.backward()
    z_opt -= gradiente  # Adam(lr=0.5)
decode(z_opt) → progresión enriquecida
```

El término de regularización `MSE(z_opt, z_orig)` evita que z se aleje demasiado.

### Per-dimension control

Cada dimensión PCS tiene su propio flag:

```
--7c 0.8      # sube 7C al 80% del máximo posible
--vnspc 0.5   # sube VNSPC al 50%
--dtmcvi 0.3
--vdr 1.0
```

Target = `c_init + strength · (1.0 - c_init)`.

### Resultados típicos

```
In:  C | Am | F | G | C
Enr: Cmaj7 | Gm7 | Gmaj7 | Dmaj7 | Cmaj7
7C:   0.006 → 0.984     7ths: 0/5 → 5/5
```

Cada dimensión tiene un efecto distinto:
- **7C**: agrega séptimas (efecto más directo)
- **VNSPC**: cambia densidad de pitch classes, modifica más la estructura rítmico-armónica
- **DTMCVI**: afecta diversidad intervalar, puede generar acordes más variados
- **VDR**: aumenta disonancia vertical, tiende a agregar tensiones

---

## 8. Pipeline de Inferencia Completa

La inferencia completa del modelo para generar progresiones con control de complejidad sigue 4 etapas:

```
┌──────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  Input   │ ──→ │   Encoder    │ ──→ │ Gradient Ascent  │ ──→ │ Decoder  │
│ "Cm G Cm"│     │ Bidir LSTM   │     │ sobre z (90 st)  │     │ LSTM     │
│ 48-dim   │     │ → z (32-dim) │     │ vía c_predictor  │     │ auto-reg │
└──────────┘     └──────────────┘     └──────────────────┘     └──────────┘
                                              │
                                              ▼
                                     loss = MSE(PCS(c_pred), target)
                                          + 0.005·MSE(z, z_orig)
```

### Etapa 1: Parse y Encode

```python
from chords.chord_encoder import progression_to_encoding

chords = ['Cmin', 'Gmaj', 'Cmin', 'Fmin', 'Cmin']  # notación paper
seq = progression_to_encoding(chords)                 # (T, 48) float32
```

### Etapa 2: Encoder → z

```python
seq_batch = seq.unsqueeze(0).to(device)               # (1, T, 48)
lengths = torch.tensor([T], device=device)
mu, logvar = model.encoder(seq_batch, lengths)        # Bidir LSTM 256×2
z = model.reparameterize(mu, logvar)                  # (1, 32)
```

### Etapa 3: Gradient Ascent sobre z

El corazón de la inferencia. Optimiza `z` para que el `c_predictor` prediga el PCS deseado:

```python
z_opt = z_orig.clone().detach().requires_grad_(True)
target_pcs = 0.5                                      # PCS objetivo

for step in range(90):
    c_pred = model.c_predictor(z_opt)[0]              # (4,) → 7C,VNSPC,DTMCVI,VDR
    pcs_pred = pcs_fn(c_pred)                          # escalar
    loss = MSE(pcs_pred, target)                       # loss de complejidad
    loss += 0.005 * MSE(z_opt, z_orig)                # regularización
    loss.backward()
    clip_grad_norm_([z_opt], 1.0)
    opt.step()                                         # Adam, lr=0.5
```

El gradiente fluye desde el PCS target a través del `c_predictor` hacia `z`. La regularización (`0.005·MSE`) mantiene `z` anclado a la progresión original, preservando estructura y tonalidad.

### Etapa 4: Decoder → Progresión

```python
gen_seq = model.generate(z_enriched, cond, max_len=T, device=device)  # auto-reg LSTM
chords = [decode_chord(gen_seq[0, t].cpu().numpy()) for t in range(T)]
# → ['Cmaj7', 'G7', 'Cmaj7', 'Fm7', 'Cmaj7']
```

### Por qué z_only_decoder=True

El decoder recibe `z` directamente, **no la condición `C`**. Esto significa que el control de complejidad no se logra seteando `C=[0.5]*4` como input al decoder (eso no tendría efecto), sino **moviendo `z` en el espacio latente** mediante gradient ascent hasta que `c_predictor(z)` alcance el target deseado.

```
C-control directo (NO funciona):    z fijo, C variable → decoder ignora C → sin efecto
Enrichment por gradiente (SÍ):      z optimizado → c_predictor(z) ≈ target → decoder usa z → progresión enriquecida
```

---

## 9. PCS: Dos Variantes (v1 y v2)

El modelo permite dos formulaciones del Perceptual Complexity Score, que difieren en **cómo el gradient ascent pondera cada dimensión** en la función de pérdida.

### v1 — Ponderado (weighted)

```
PCS_w = (0.443·7C + 0.378·VNSPC + 0.338·DTMCVI + 0.201·VDR) / 1.360
```

- Pesos derivados del estudio perceptual con 43 participantes (p < 0.05)
- **7C pesa 2.2× más que VDR** (0.443 vs 0.201)
- El gradiente `∂loss/∂c` empuja más fuerte hacia 7C y VNSPC
- **Efecto:** para alcanzar PCS=0.5, el optimizer prioriza agregar séptimas
- Resultado: más 7th chords, menos cambios en otras dimensiones

### v2 — Promediado (equal)

```
PCS_eq = (7C + VNSPC + DTMCVI + VDR) / 4
```

- Las 4 dimensiones pesan 0.25 cada una
- El gradiente `∂loss/∂c` es uniforme: empuja todas por igual
- **Efecto:** para alcanzar PCS=0.5, el optimizer distribuye el cambio entre las 4 dimensiones
- Resultado: menos séptimas, más variabilidad armónica (sus, dim, no3d) y disonancia

### Comparación

| Métrica | v1 (weighted) | v2 (equal) |
|---------|--------------|------------|
| Origen | Estudio perceptual | Agnóstico |
| Prioridad | 7C > VNSPC > DTMCVI > VDR | Todas igual |
| 7th chords generados | Alta (hasta 5/5) | Moderada (1-2/5) |
| Variedad armónica | Enfocada en séptimas | Distribuida (sus, dim, no3d) |
| Coherencia tonal | 88% roots en escala | 88% roots en escala |
| Uso | Complejidad validada perceptual | Comparable con baselines |

### Modo por dimensión individual

También se puede enriquecer una sola dimensión sin tocar las demás:

```python
loss = MSE(c_pred[dim_idx], target)   # solo 7C, solo VDR, etc.
```

Esto permite control fino: "solo quiero más séptimas", "solo quiero más disonancia", etc.

---

## 10. Experimentos y Evaluación

Los experimentos están en `experiments/complexity_comparison.ipynb`. Comparan SCL 2.0 (v1 y v2) contra CVAE y RVAE de referencia, generando 5 niveles de complejidad a partir de una progresión inicial simple.

### Setup experimental

| Parámetro | Valor |
|-----------|-------|
| Progresión inicial | `Cmin \| Gmaj \| Cmin \| Fmin \| Cmin` |
| Checkpoint | `rvae_key_v13/best.pt` (latent=32, epoch=30) |
| Niveles de complejidad | 5 (target PCS: 0.00, 0.25, 0.50, 0.75, 1.00) |
| Gradient ascent | Adam, lr=0.5, 90 steps, reg=0.005 |
| Baselines | CVAE y RVAE (resultados de paper externo) |

### Métricas evaluadas

| Métrica | Descripción |
|---------|-------------|
| **PCS alcanzado** | Qué tan cerca llega del target |
| **Δ por dimensión** | Cuánto cambia 7C, VNSPC, DTMCVI, VDR |
| **Ratio Δ7C/ΣΔothers** | Qué dimensión domina el enrichment |
| **% séptimas generadas** | Proporción de acordes con séptima |
| **Tipos de acorde** | Distribución (tríada, 7th, sus, dim, no3d) |
| **Coherencia tonal** | % roots en escala, overlap posicional, key match |
| **Desplazamiento \|\|z\|\|** | Qué tanto se aleja z del original |

### Resultados clave

**Convergencia del gradient ascent:**

Ambas versiones (v1 y v2) convergen al target PCS en ~20-30 steps de los 90 totales. La trayectoria es similar pero llegan desde direcciones distintas en el espacio latente.

**Distribución del gradiente (ratio Δ7C / ΣΔothers):**

| Nivel | Target | v1 ratio | v2 ratio |
|-------|--------|----------|----------|
| 2 | 0.25 | 0.19 | 0.00 |
| 3 | 0.50 | 0.11 | 0.08 |
| 4 | 0.75 | **0.57** | 0.15 |
| 5 | 1.00 | **0.51** | 0.00 |

En niveles altos v1 depende ~50-60% de 7C para alcanzar el target, mientras v2 apenas la usa (0-15%) y depende de VNSPC/DTMCVI/VDR.

**Coherencia tonal (promedio 5 niveles):**

| Modelo | % en escala | % overlap | % cualidad |
|--------|------------|-----------|------------|
| CVAE (paper) | 84% | **84%** | **72%** |
| RVAE (paper) | 76% | 64% | 60% |
| **SCL v1 (weighted)** | **88%** | 32% | 32% |
| **SCL v2 (equal)** | **88%** | 36% | 44% |

SCL 2.0 tiene la **mejor adherencia a la escala** (88%) a pesar de explorar mucho más el espacio de acordes (overlap bajo). El `key_predictor` entrenado end-to-end ayuda a mantener la tonalidad incluso cuando `z` se aleja del original.

**Desplazamiento en el espacio latente:**

A mayor target PCS, mayor \|\|z_enriched − z_orig\|\|. Ambas versiones mueven `z` magnitudes similares (~4-7 unidades para PCS=1.0, con \|\|z_orig\|\| ≈ 8-10) pero en direcciones distintas, resultando en perfiles armónicos diferentes.

---

## 11. Scripts / Cheatsheet

### Preparación de datos

```bash
# 1. Extraer features con jSymbolic
java -jar jSymbolic_2_2_user/jSymbolic2.jar \
  -config Dataset/jsymbolic_config.txt \
  -output Dataset/dataset_with_jsymbolic.parquet

# 2. Generar dataset simbólico (MIDI → 48-dim vectors)
Dataset/generate_dataset_with_jsymbolic.py

# 3. Crear dataset condicionado (unir symbolic + jSymbolic + PCS)
.venv/bin/python cvae/prepare_conditioning.py
```

### Entrenamiento

```bash
# Default
.venv/bin/python scripts/train_rvae.py

# Configuración usada para v13
.venv/bin/python scripts/train_rvae.py \
  --parquet Dataset/dataset_conditioned.parquet \
  --cond-cols 7C VNSPC DTMCVI VDR --condition-dim 4 \
  --latent-dim 128 --epochs 30 --beta 0.01 \
  --kl-cycle 10 --free-bits 1.0 --per-dim-free-bits 0.25 \
  --tf-start 1.0 --tf-end 0.3 --tf-epochs 15 \
  --word-dropout 0.7 --checkpoint-dir checkpoints/rvae_key_v13

# Resumir desde checkpoint
--resume checkpoints/rvae_key_v13/last.pt

# Con mixed precision
--amp
```

### Enrichment

```bash
# Gradiente ascent básico (7C default=1.0)
.venv/bin/python scripts/enrich.py --chords "C | Am | F | G | C"

# Mostrar reconstrucción original + targets específicos
.venv/bin/python scripts/enrich.py --chords "C | F | G | C" \
  --7c 0.5 --dtmcvi 0.3 --show-chords

# Solo reconstruir (strengths = 0)
.venv/bin/python scripts/enrich.py --chords "C | F | G7 | Am" \
  --7c 0 --vnspc 0 --dtmcvi 0 --vdr 0

# Probar todas las dimensiones en dataset
.venv/bin/python scripts/enrich_gradient.py --example 0 --all-dims

# Probar varios ejemplos
.venv/bin/python scripts/enrich_gradient.py --examples 0 5 10 --all-dims
```

### Generación

```bash
# Interpolar complejidad (z fijo, varía C)
.venv/bin/python scripts/generate_rvae.py checkpoints/rvae_key_v13/best.pt \
  --interpolate --example-idx 0 \
  --latent-dim 32 --condition-dim 16

# Samplear del prior con C específico
.venv/bin/python scripts/generate_rvae.py checkpoints/rvae_key_v13/best.pt \
  --sample-prior --c-values 0.9 0.5 0.5 0.5 \
  --num-samples 2 --gen-length 8 --key 0 \
  --latent-dim 32 --condition-dim 16

# Enriquecer por interpolación z (alpha blend)
.venv/bin/python scripts/generate_rvae.py checkpoints/rvae_key_v13/best.pt \
  --multipliers 1.5 1.0 1.0 1.0 --alpha 0.8 \
  --latent-dim 32 --condition-dim 16

# Enriquecer con vector heurístico precalculado
.venv/bin/python scripts/generate_rvae.py checkpoints/rvae_key_v13/best.pt \
  --heuristic-vector checkpoints/rvae_key_v2/enrichment_vector.pt \
  --alpha 2.0 --multipliers 1.5 1.0 1.0 1.0 \
  --latent-dim 32 --condition-dim 16
```

### Evaluación

```bash
.venv/bin/python scripts/evaluate_rvae_v11_mem_safe.py
```

### Sweep de hiperparámetros

```bash
.venv/bin/python scripts/sweep_rvae.py
```

### Checkpoints

| Archivo | Latent | Cond | Descripción |
|---------|--------|------|-------------|
| `checkpoints/rvae_key_v13/best.pt` | 32 | 16 (4 perc + 12 key) | z_only_decoder, val_loss=4.2376 |
| `checkpoints/rvae_key_v13/last.pt` | 32 | 16 | Epoch 30 |

---

## Estructura del proyecto

```
SCL_2.0/
├── Dataset/                   # Parquets (symbolic, conditioned, jsymbolic)
├── PCS/                       # Perceptual Complexity Score
│   ├── pcs.py                 #   Cálculo de PCS
│   ├── pcs_weights.json       #   Pesos de las 4 dimensiones
│   └── analisis_msi_export.ipynb  #   Análisis del estudio perceptual
├── chords/                    # Codificación de acordes
│   ├── vocab.py               #   Vocabulario (48-dim)
│   ├── chord_encoder.py       #   Parseo, encode, decode
│   └── dataset_converter.py   #   MIDI → secuencias simbólicas
├── cvae/                      # Modelo
│   ├── config.py              #   Hiperparámetros
│   ├── dataset.py             #   Dataset + dataloader
│   ├── models/rvae.py         #   RVAE completo
│   ├── losses.py              #   Pérdidas auxiliares (coh, tens, mov)
│   └── prepare_conditioning.py #  Unir symbolic + jSymbolic + PCS
├── experiments/               # Experimentos y comparaciones
│   ├── complexity_comparison.ipynb          # Comparación CVAE/RVAE/SCL
│   └── complexity_comparison_executed.ipynb # Ejecutado con gráficos
├── scripts/                   # Scripts de entrada
│   ├── train_rvae.py          #   Entrenamiento
│   ├── enrich.py              #   Enrichment por gradient ascent
│   ├── enrich_gradient.py     #   Enrichment batch sobre dataset
│   ├── generate_rvae.py       #   Generación + interpolación
│   ├── evaluate_rvae_v11_mem_safe.py  # Evaluación post-entrenamiento
│   └── sweep_rvae.py          #   Sweep de hiperparámetros
├── checkpoints/               # Modelos entrenados
│   └── rvae_key_v13/          #   best.pt, last.pt
├── runs/                      # TensorBoard logs
├── notebooks/                 # Notebooks de análisis
└── jSymbolic_2_2_user/        # jSymbolic jar + recursos
```
