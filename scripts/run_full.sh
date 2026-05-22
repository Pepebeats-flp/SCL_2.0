#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

RUN_NAME="cvae_pcs_full"
mkdir -p "checkpoints/${RUN_NAME}" "runs/${RUN_NAME}"

nohup python -u scripts/train.py \
    --condition-dim 5 \
    --lr 5e-4 \
    --epochs 40 \
    --patience 7 \
    --batch-size 128 \
    --free-bits 1.5 \
    --beta 1.0 \
    --word-dropout 0.3 \
    --kl-warmup 10 \
    --checkpoint-dir "checkpoints/${RUN_NAME}" \
    --log-dir "runs/${RUN_NAME}" \
    > "training_${RUN_NAME}.log" 2>&1 &

echo "=== Training launched ==="
echo "  PID:      $!"
echo "  Log:      training_${RUN_NAME}.log"
echo "  Monitor:  tail -f training_${RUN_NAME}.log"
echo "  TensorBoard: tensorboard --logdir runs/${RUN_NAME}"
echo ""
echo "Quality guards active:"
echo "  - KL collapse (KL<0.001 for 2 epochs)"
echo "  - Recon cheating (recon<0.001 with KL~0)"
echo "  - Coherence worsening (3+ epochs increasing)"
echo "  - Early stopping patience (7 epochs)"
echo ""
echo "After training, evaluate on held-out test set (1000 samples):"
echo "  python scripts/evaluate.py checkpoints/${RUN_NAME}/best.pt --output test_results.json"
echo ""
echo "If stopped early: adjust hyperparams, delete checkpoints/${RUN_NAME}/crash_*.pt, and re-run."
