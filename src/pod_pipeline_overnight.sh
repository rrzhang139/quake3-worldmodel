#!/bin/bash
# pod_pipeline_overnight.sh — Full overnight pipeline with smart resume
#
# Stages (each is skipped if already done):
#   1. Install deps
#   2. Download episodes from HF (skip if already on disk)
#   3. Encode latents via SD VAE (skip if already encoded)
#   4. Upload latents to HF (skip if already uploaded)
#   5. Train latent diffusion — medium model, 60 epochs
#   6. Upload best.pt to W&B artifacts
#   7. Self-terminate pod
#
# Secrets via env vars: HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN, RUNPOD_API_KEY
# (pre-loaded from /workspace/.env or container env)

set -e
LOG=/workspace/pipeline_overnight.log
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "=== Overnight Pipeline start $(date) ==="
echo "Pod ID: ${RUNPOD_POD_ID:-cyzx60pwcdht7t}"
echo "============================================"

# Load .env if exists (secrets may also come from container env)
[ -f /workspace/.env ] && set -a && source /workspace/.env && set +a

# ─── STAGE 1: INSTALL DEPS ───────────────────────────────────────────────────
echo ""
echo "=== [1/7] Installing deps ==="
cd /workspace/quake3-worldmodel
python3 -m venv .venv
source .venv/bin/activate

pip install --quiet --no-warn-script-location \
    diffusers transformers accelerate \
    wandb huggingface_hub scikit-image \
    imageio imageio-ffmpeg
echo "Deps installed. Python: $(python --version), Torch: $(python -c 'import torch; print(torch.__version__)')"

# Auth
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"

# ─── STAGE 2: DOWNLOAD LATENTS DIRECTLY FROM HF (fast path) ─────────────────
# Latents are pre-encoded and already on HF — download them directly.
# This skips ~19GB episode download + ~40min GPU encoding.
LATENT_DIR=/workspace/data/latents_160x120_5k
LATENT_COUNT=$(ls "$LATENT_DIR"/episode_*.pt 2>/dev/null | wc -l || echo 0)
echo ""
echo "=== [2/7] Latents: $LATENT_COUNT already on disk ==="

if [ "$LATENT_COUNT" -lt 4900 ]; then
    echo "Downloading latents_160x120_5k from HF..."
    mkdir -p "$LATENT_DIR"
    python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="rzhang139/vizdoom-episodes",
    repo_type="dataset",
    local_dir="/workspace/data",
    allow_patterns=["latents_160x120_5k/*"],
    ignore_patterns=["*.md", ".gitattributes"],
)
import glob
count = len(glob.glob("/workspace/data/latents_160x120_5k/episode_*.pt"))
print(f"Download complete: {count} latents")
PYEOF
else
    echo "Skipping download — $LATENT_COUNT latents already present"
fi

# ─── STAGE 3: ENCODE LATENTS (skip — already downloaded) ─────────────────────
LATENT_COUNT=$(ls "$LATENT_DIR"/episode_*.pt 2>/dev/null | wc -l || echo 0)
echo ""
echo "=== [3/7] Latents: $LATENT_COUNT on disk (encoding skipped — using pre-encoded from HF) ==="

# ─── STAGE 4: UPLOAD LATENTS TO HF ──────────────────────────────────────────
echo ""
echo "=== [4/7] Uploading latents to HuggingFace ==="
python3 - << 'PYEOF'
from huggingface_hub import HfApi, list_repo_files
api = HfApi()

# Check how many already uploaded
try:
    uploaded = sum(1 for f in list_repo_files("rzhang139/vizdoom-episodes", repo_type="dataset")
                   if f.startswith("latents_160x120_5k/episode_"))
except Exception:
    uploaded = 0

import glob, os
local = sorted(glob.glob("/workspace/data/latents_160x120_5k/episode_*.pt"))
print(f"Local: {len(local)} latents, HF: {uploaded} already uploaded")

if uploaded >= len(local) - 10:
    print("Already uploaded, skipping.")
else:
    print(f"Uploading {len(local)} latent files to HF...")
    api.upload_folder(
        folder_path="/workspace/data/latents_160x120_5k",
        repo_id="rzhang139/vizdoom-episodes",
        repo_type="dataset",
        path_in_repo="latents_160x120_5k",
        allow_patterns=["episode_*.pt"],
    )
    print("Upload complete!")
PYEOF

# ─── STAGE 5: TRAIN ──────────────────────────────────────────────────────────
echo ""
echo "=== [5/7] Training medium latent diffusion model (60 epochs) ==="
echo "Model: medium (~12M params), batch=128, noise_aug=0.3, LR warmup 200 steps"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/workspace/experiments/run_latent_medium
mkdir -p "$OUT"

python3 -u src/train_latent.py \
    --data "$LATENT_DIR" \
    --epochs 60 \
    --batch_size 128 \
    --model_size medium \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --lr_warmup_steps 200 \
    --noise_aug 0.3 \
    --cfg_drop_prob 0.15 \
    --action_aux_weight 0.1 \
    --max_grad_norm 1.0 \
    --wandb \
    --output "$OUT" \
    --log_every 200 \
    --save_every 10 \
    2>&1 | tee /workspace/train_latent_medium.log

echo "Training complete."

# ─── STAGE 6: UPLOAD CHECKPOINT TO W&B ──────────────────────────────────────
echo ""
echo "=== [6/7] Uploading checkpoint to W&B ==="
python3 - << 'PYEOF'
import wandb, os
run = wandb.init(
    project="quake3-worldmodel", entity="rzhang139",
    job_type="artifact-upload",
    name="upload-latent-medium-60ep"
)
artifact = wandb.Artifact(
    "quake3-wm-latent-medium-60ep", type="model",
    description="Latent diffusion medium model (~12M params), 60 epochs, 5K eps, noise_aug=0.3"
)
artifact.add_file("/workspace/experiments/run_latent_medium/best.pt")
run.log_artifact(artifact)
run.finish()
print("Checkpoint uploaded: quake3-wm-latent-medium-60ep")
PYEOF

# ─── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "=== All done! $(date) ==="
echo "Best checkpoint: /workspace/experiments/run_latent_medium/best.pt"
echo "W&B artifact: quake3-wm-latent-medium-60ep"
echo "============================================"

# Stage 7: Self-terminate
echo "=== [7/7] Self-terminating pod ==="
POD_ID="${RUNPOD_POD_ID:-cyzx60pwcdht7t}"
curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"${POD_ID}\\\" }) }\"}" \
    && echo "Pod termination requested." || echo "WARNING: Self-terminate failed — stop pod manually!"
