#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

SESSION_NAME="rvae_training"
RUN_NAME="rvae_100k_4dim"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "La sesion tmux '$SESSION_NAME' ya existe."
    echo ""
    echo "  Para adjuntarte:"
    echo "    tmux attach -t $SESSION_NAME"
    echo ""
    echo "  Para ver el log:"
    echo "    tail -f training_${RUN_NAME}.log"
    echo ""
    echo "  Para matar la sesion (solo si ya termino):"
    echo "    tmux kill-session -t $SESSION_NAME"
    exit 1
fi

mkdir -p "checkpoints/${RUN_NAME}" "runs/${RUN_NAME}"

tmux new-session -d -s "$SESSION_NAME"

tmux send-keys -t "$SESSION_NAME" \
    "source .venv/bin/activate && \
     python -u scripts/train_rvae.py \
       --parquet Dataset/dataset_conditioned_100k.parquet \
        --condition-dim 4 \
        --latent-dim 8 \
        --cond-cols 7C VNSPC DTMCVI VDR \
       --lr 5e-4 \
       --epochs 30 \
       --patience 10 \
       --batch-size 256 \
       --num-workers 0 \
        --free-bits 0.0 \
        --per-dim-free-bits 0.25 \
        --word-dropout 0.9 \
       --beta 1.0 \
       --kl-warmup 15 \
       --kl-cycle 0 \
       --lambda-coh 0.0 \
       --lambda-tens 0.0 \
       --lambda-mov 0.0 \
       --tf-start 1.0 \
       --tf-end 1.0 \
       --tf-epochs 1 \
       --active-units-threshold 0.05 \
       --kl-real-threshold 0.001 \
       --amp \
       --checkpoint-dir checkpoints/${RUN_NAME} \
       --log-dir runs/${RUN_NAME} \
       2>&1 | tee training_${RUN_NAME}.log" \
    "Enter"

echo "=== RVAE entrenamiento lanzado en tmux ==="
echo "  Sesion:   $SESSION_NAME"
echo "  PID tmux: $(tmux list-sessions 2>/dev/null)"
echo "  Log:      training_${RUN_NAME}.log"
echo ""
echo "COMANDOS:"
echo "  Adjuntar:         tmux attach -t $SESSION_NAME"
echo "  Desadjuntar:      Ctrl+B, luego D"
echo "  Ver log afuera:   tail -f training_${RUN_NAME}.log"
echo "  TensorBoard:      tensorboard --logdir runs/${RUN_NAME}"
echo "  Matar sesion:     tmux kill-session -t $SESSION_NAME"
echo ""
echo "=== Hyperparams RVAE ==="
echo "  - Encoder sin C (X -> q(z|X))"
echo "  - ComplexityPrior: MLP aprende p(z|C) -> p(z|C) = N(mu, sigma)"
echo "  - KL(q(z|X) || p(z|C)) organiza latente por complejidad"
echo "  - 4 dims perceptuales: 7C, VNSPC, DTMCVI, VDR"
echo "  - TF: 1.0 (teacher forcing) + word_dropout=0.9 (90% inputs masked)"
echo "  - KL warmup: 15 epochs (lineal 0->1.0), sin ciclo"
echo "  - 100k subset (test rapido)"
echo ""
echo "Despues del entrenamiento, generar:"
echo "  python scripts/generate_rvae.py checkpoints/${RUN_NAME}/best.pt --num-examples 5"
