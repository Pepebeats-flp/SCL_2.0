#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

SESSION_NAME="rvae_full"
RUN_NAME="rvae_full_4dim"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "La sesion tmux '$SESSION_NAME' ya existe."
    echo "  Adjuntar: tmux attach -t $SESSION_NAME"
    echo "  Log: tail -f training_${RUN_NAME}.log"
    echo "  Matar: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

mkdir -p "checkpoints/${RUN_NAME}" "runs/${RUN_NAME}"

tmux new-session -d -s "$SESSION_NAME"

tmux send-keys -t "$SESSION_NAME" \
    "source .venv/bin/activate && \
     python -u scripts/train_rvae.py \
       --parquet Dataset/dataset_conditioned.parquet \
       --condition-dim 4 \
       --cond-cols 7C VNSPC DTMCVI VDR \
       --lr 5e-4 \
       --epochs 50 \
       --patience 15 \
       --batch-size 256 \
       --num-workers 8 \
       --free-bits 1.0 \
       --word-dropout 0.7 \
       --beta 0.01 \
       --kl-warmup 2 \
       --kl-cycle 10 \
       --lambda-coh 0.0 \
       --lambda-tens 0.0 \
       --lambda-mov 0.0 \
       --tf-start 1.0 \
       --tf-end 0.3 \
       --tf-epochs 15 \
       --active-units-threshold 0.15 \
       --kl-real-threshold 0.001 \
       --amp \
       --checkpoint-dir checkpoints/${RUN_NAME} \
       --log-dir runs/${RUN_NAME} \
       2>&1 | tee training_${RUN_NAME}.log" \
    "Enter"

echo "=== RVAE FULL DATASET lanzado en tmux ==="
echo "  Sesion:   $SESSION_NAME"
echo "  Log:      training_${RUN_NAME}.log"
echo ""
echo "COMANDOS:"
echo "  Adjuntar:         tmux attach -t $SESSION_NAME"
echo "  Ver log:          tail -f training_${RUN_NAME}.log"
echo "  TensorBoard:      tensorboard --logdir runs/${RUN_NAME}"
echo "  Matar sesion:     tmux kill-session -t $SESSION_NAME"
echo ""
echo "=== Hyperparams ==="
echo "  - 877k progressions (full dataset)"
echo "  - 4 dims: 7C, VNSPC, DTMCVI, VDR (sin PCS)"
echo "  - TF annealing: 1.0 -> 0.3 over 15 epochs"
echo "  - KL cycle: 10 | Beta target: 0.01"
echo "  - Epochs: 50 | Patience: 15"
echo "  - Batch: 256 | Workers: 8 | AMP: on"
