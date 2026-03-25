#!/bin/bash
# pod_pipeline_v2.sh — Scale-up: 50K episodes, large model, auto-tuned batch size
#
# Stages:
#   1. Install deps (vizdoom + training stack)
#   2. Collect 50K episodes via ViZDoom (skip if already on disk)
#   3. Encode episodes to latents via SD VAE (skip if already encoded)
#   4. Upload latents to HF as latents_160x120_50k (skip if already done)
#   5. Auto-tune batch size to 80-90% GPU VRAM utilization
#   6. Train large model with optimal batch + scaled LR
#   7. Upload best.pt to W&B artifacts
#   8. Self-terminate pod
#
# Secrets: HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY (from /workspace/.env)

set -e
LOG=/workspace/pipeline_v2.log
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "=== Pipeline v2 start $(date) ==="
echo "Pod ID: ${RUNPOD_POD_ID:-unknown}"
echo "============================================"

[ -f /workspace/.env ] && set -a && source /workspace/.env && set +a

REPO_DIR=/workspace/quake3-worldmodel
EP_DIR=/workspace/data/episodes_160x120_50k
LATENT_DIR=/workspace/data/latents_160x120_50k
OUT=/workspace/experiments/run_large_50k

# ─── STAGE 1: INSTALL DEPS ───────────────────────────────────────────────────
echo ""
echo "=== [1/8] Installing deps ==="
cd "$REPO_DIR"
# --system-site-packages: inherit torch from RunPod base image (already cu124)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install --quiet --no-warn-script-location \
    vizdoom \
    diffusers transformers accelerate \
    wandb huggingface_hub scikit-image \
    imageio imageio-ffmpeg

# Verify CUDA
python3 -c "
import torch, sys
ok = torch.cuda.is_available()
print(f'Torch: {torch.__version__}, CUDA: {ok}')
if not ok:
    print('ERROR: CUDA not available — aborting')
    sys.exit(1)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB')
"

python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
echo "Deps OK."

# ─── STAGE 2: COLLECT 50K EPISODES ───────────────────────────────────────────
echo ""
EP_COUNT=$(ls "$EP_DIR"/episode_*.pt 2>/dev/null | wc -l || echo 0)
echo "=== [2/8] Episodes: $EP_COUNT already on disk ==="

if [ "$EP_COUNT" -lt 49000 ]; then
    echo "Collecting 50K episodes (mixed policy, 160x120)..."
    mkdir -p "$EP_DIR"
    python3 -u src/collect.py \
        --num_episodes 50000 \
        --output "$EP_DIR" \
        --max_steps 300 \
        --policy mixed \
        --res 0 \
        --screen_res 160x120
    EP_COUNT=$(ls "$EP_DIR"/episode_*.pt | wc -l)
    echo "Collection done: $EP_COUNT episodes"
else
    echo "Skipping — $EP_COUNT episodes already present"
fi

# ─── STAGE 3: ENCODE LATENTS ──────────────────────────────────────────────────
echo ""
LATENT_COUNT=$(ls "$LATENT_DIR"/episode_*.pt 2>/dev/null | wc -l || echo 0)
echo "=== [3/8] Latents: $LATENT_COUNT already encoded ==="

if [ "$LATENT_COUNT" -lt 49000 ]; then
    echo "Encoding episodes → latents via SD VAE..."
    mkdir -p "$LATENT_DIR"
    python3 -u src/encode_latents.py \
        --input "$EP_DIR" \
        --output "$LATENT_DIR" \
        --batch_size 128
    LATENT_COUNT=$(ls "$LATENT_DIR"/episode_*.pt | wc -l)
    echo "Encoding done: $LATENT_COUNT latents"
else
    echo "Skipping — $LATENT_COUNT latents already encoded"
fi

# ─── STAGE 4: UPLOAD LATENTS TO HF (background — runs in parallel with training) ──
echo ""
echo "=== [4/8] Starting HF upload in background (won't block training) ==="
python3 - << 'PYEOF' &
HF_UPLOAD_PID=$!
from huggingface_hub import HfApi, list_repo_files
import glob, sys

api = HfApi()
local = sorted(glob.glob("/workspace/data/latents_160x120_50k/episode_*.pt"))
print(f"[HF Upload] Local latents: {len(local)}", flush=True)

try:
    uploaded = sum(1 for f in list_repo_files("rzhang139/vizdoom-episodes", repo_type="dataset")
                   if f.startswith("latents_160x120_50k/episode_"))
except Exception:
    uploaded = 0

print(f"[HF Upload] Already on HF: {uploaded}", flush=True)
if uploaded >= len(local) - 100:
    print("[HF Upload] Already uploaded, skipping.", flush=True)
else:
    print(f"[HF Upload] Uploading {len(local)} files...", flush=True)
    api.upload_folder(
        folder_path="/workspace/data/latents_160x120_50k",
        repo_id="rzhang139/vizdoom-episodes",
        repo_type="dataset",
        path_in_repo="latents_160x120_50k",
        allow_patterns=["episode_*.pt"],
    )
    print("[HF Upload] Done.", flush=True)
PYEOF
echo "HF upload running in background (PID: $HF_UPLOAD_PID). Training starts immediately."

# ─── STAGE 5: AUTO-TUNE BATCH SIZE ───────────────────────────────────────────
echo ""
echo "=== [5/8] Auto-tuning batch size for 80-90% VRAM ==="
OPTIMAL_BATCH=$(python3 - << 'PYEOF'
import torch, sys
sys.path.insert(0, "src")
from model import make_denoiser

device = "cuda"
vram_total = torch.cuda.get_device_properties(0).total_memory
target_lo = 0.80
target_hi = 0.90

model = make_denoiser(
    num_actions=10, img_size=20, img_channels=4,
    num_context_frames=4, model_size="large",
    cfg_drop_prob=0.15, action_aux_weight=0.1,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

best_batch = 64
for batch in [64, 128, 256, 512, 1024, 1536, 2048]:
    try:
        torch.cuda.empty_cache()
        # Simulate a real training step
        ctx = torch.randn(batch, 4, 4, 15, 20, device=device)
        act = torch.randint(0, 10, (batch,), device=device)
        # Sample sigma from log-normal (same as train_latent.py)
        sigma = (torch.randn(batch, device=device) * 1.2 - 0.4).exp()
        optimizer.zero_grad()
        loss = model.training_loss(ctx, act, sigma)
        loss.backward()
        optimizer.step()

        used = torch.cuda.max_memory_allocated()
        util = used / vram_total
        print(f"  batch={batch:5d}: {used/1e9:.1f}GB / {vram_total/1e9:.1f}GB = {util*100:.0f}%", flush=True)
        torch.cuda.reset_peak_memory_stats()

        if target_lo <= util <= target_hi:
            best_batch = batch
            break
        elif util < target_lo:
            best_batch = batch  # keep going up
        else:
            # Overshot — use previous batch
            break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  batch={batch:5d}: OOM", flush=True)
        break

print(f"OPTIMAL_BATCH={best_batch}")
PYEOF
)
BATCH_SIZE=$(echo "$OPTIMAL_BATCH" | grep "OPTIMAL_BATCH=" | cut -d= -f2)
echo "Selected batch size: $BATCH_SIZE"

# Linear LR scaling: base LR=1e-4 at batch=128; scale proportionally
LR=$(python3 -c "print(f'{1e-4 * int(\"$BATCH_SIZE\") / 128:.2e}')")
WARMUP=$(python3 -c "print(int(200 * int(\"$BATCH_SIZE\") / 128))")
echo "Scaled LR: $LR  Warmup steps: $WARMUP"

# ─── STAGE 6: TRAIN ──────────────────────────────────────────────────────────
echo ""
echo "=== [6/8] Training large model (30 epochs, 50K episodes) ==="
echo "Batch=$BATCH_SIZE  LR=$LR  Warmup=$WARMUP  noise_aug=0.5"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

python3 -u src/train_latent.py \
    --data "$LATENT_DIR" \
    --epochs 30 \
    --batch_size "$BATCH_SIZE" \
    --model_size large \
    --lr "$LR" \
    --weight_decay 1e-2 \
    --lr_warmup_steps "$WARMUP" \
    --noise_aug 0.5 \
    --cfg_drop_prob 0.15 \
    --action_aux_weight 0.1 \
    --max_grad_norm 1.0 \
    --bf16 \
    --compile \
    --wandb \
    --output "$OUT" \
    --log_every 500 \
    --save_every 5 \
    2>&1 | tee /workspace/train_large.log

echo "Training complete."

# Wait for HF upload to finish (likely already done)
wait $HF_UPLOAD_PID 2>/dev/null && echo "HF upload complete." || echo "HF upload may have finished already."

# ─── STAGE 7: UPLOAD CHECKPOINT ──────────────────────────────────────────────
echo ""
echo "=== [7/8] Uploading checkpoint to W&B ==="
python3 - << 'PYEOF'
import wandb, os
run = wandb.init(
    project="quake3-worldmodel", entity="rzhang139",
    job_type="artifact-upload",
    name="upload-large-50k-30ep"
)
artifact = wandb.Artifact(
    "quake3-wm-large-50k-30ep", type="model",
    description="Large model (~21M params), 30 epochs, 50K eps, noise_aug=0.5, batch auto-tuned to 80-90% VRAM"
)
artifact.add_file("/workspace/experiments/run_large_50k/best.pt")
run.log_artifact(artifact)
run.finish()
print("Uploaded: quake3-wm-large-50k-30ep")
PYEOF

echo ""
echo "============================================"
echo "=== All done! $(date) ==="
echo "============================================"

# ─── STAGE 8: SELF-TERMINATE ─────────────────────────────────────────────────
echo "=== [8/8] Self-terminating pod ==="
POD_ID="${RUNPOD_POD_ID:-unknown}"
curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"${POD_ID}\\\" }) }\"}" \
    && echo "Pod termination requested." || echo "WARNING: Self-terminate failed — stop pod manually!"
