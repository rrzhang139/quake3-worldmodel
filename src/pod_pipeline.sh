#!/bin/bash
# Full pod pipeline: encode latents + train latent diffusion + upload
# Secrets passed via env vars: HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN, RUNPOD_API_KEY
set -e

LOG=/workspace/pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== Pipeline start $(date) ==="
echo "Pod ID: ${RUNPOD_POD_ID:-unknown}"

# Setup repo
cd /workspace
if [ ! -d quake3-worldmodel ]; then
    git clone "https://${GITHUB_TOKEN}@github.com/rrzhang139/quake3-worldmodel.git"
fi
cd /workspace/quake3-worldmodel

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "=== Installing deps ==="
pip install --quiet --no-warn-script-location torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
pip install --quiet --no-warn-script-location \
    wandb numpy pillow imageio imageio-ffmpeg \
    diffusers transformers accelerate huggingface_hub scikit-image
echo "Deps installed."

# Login
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"

# Check if latents already exist on HF
echo "=== Checking for existing latents on HF ==="
LATENT_COUNT=$(python3 - << 'PYEOF'
from huggingface_hub import list_repo_files
try:
    files = [f for f in list_repo_files("rzhang139/vizdoom-episodes", repo_type="dataset")
             if f.startswith("latents_160x120_5k/episode_")]
    print(len(files))
except Exception as e:
    print(0)
PYEOF
)
echo "Found $LATENT_COUNT latent files on HF"

# Download episodes
echo "=== Downloading episodes from HF ==="
mkdir -p /workspace/data
python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="rzhang139/vizdoom-episodes",
    repo_type="dataset",
    local_dir="/workspace/data/hf_episodes",
    allow_patterns=["episodes_160x120_5k/*"],
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Download complete")
PYEOF

EPISODES_DIR=$(find /workspace/data -name "episode_000000.pt" -maxdepth 5 | head -1 | xargs dirname)
echo "Episodes dir: $EPISODES_DIR"

# Encode latents (unless already done)
LATENT_DIR=/workspace/data/latents_160x120_5k
mkdir -p "$LATENT_DIR"
EXISTING=$(ls "$LATENT_DIR"/*.pt 2>/dev/null | wc -l || echo 0)

if [ "$EXISTING" -lt 100 ]; then
    echo "=== Encoding latents with SD VAE ==="
    python3 src/encode_latents.py \
        --input "$EPISODES_DIR" \
        --output "$LATENT_DIR" \
        --batch_size 64
else
    echo "=== Found $EXISTING existing latents, skipping encode ==="
fi

# Upload latents to HF (skip if already there)
if [ "$LATENT_COUNT" -lt 100 ]; then
    echo "=== Uploading latents to HuggingFace ==="
    python3 - << 'PYEOF'
from huggingface_hub import HfApi
import glob, os
api = HfApi()
lat_files = sorted(glob.glob("/workspace/data/latents_160x120_5k/episode_*.pt"))
print(f"Uploading {len(lat_files)} latent files...")
# Upload in batches of 50
for i in range(0, len(lat_files), 50):
    batch = lat_files[i:i+50]
    api.upload_folder(
        folder_path="/workspace/data/latents_160x120_5k",
        repo_id="rzhang139/vizdoom-episodes",
        repo_type="dataset",
        path_in_repo="latents_160x120_5k",
        allow_patterns=[os.path.basename(f) for f in batch],
    )
    print(f"  uploaded batch {i//50+1}")
print("Upload complete!")
PYEOF
else
    echo "=== Latents already on HF, skipping upload ==="
fi

# Train latent diffusion model
echo "=== Training latent diffusion model ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/experiments/run_latent_small

python3 -u src/train_latent.py \
    --data "$LATENT_DIR" \
    --epochs 30 \
    --batch_size 128 \
    --model_size small \
    --lr 1e-4 \
    --noise_aug 0.3 \
    --wandb \
    --output /workspace/experiments/run_latent_small \
    --log_every 100 \
    --save_every 5 \
    2>&1 | tee /workspace/train_latent.log

echo "=== Training complete! ==="

# Upload checkpoint to W&B
python3 - << 'PYEOF'
import wandb
run = wandb.init(project="quake3-worldmodel", entity="rzhang139", job_type="artifact-upload",
                 name="upload-latent-small-30ep")
artifact = wandb.Artifact("quake3-wm-latent-small-30ep", type="model",
                          description="Latent diffusion small model 30 epochs 5K episodes")
artifact.add_file("/workspace/experiments/run_latent_small/best.pt")
run.log_artifact(artifact)
run.finish()
print("Checkpoint uploaded to W&B!")
PYEOF

echo "=== All done! $(date) ==="

# Self-terminate pod
if [ -n "$RUNPOD_API_KEY" ] && [ -n "$RUNPOD_POD_ID" ]; then
    echo "Self-terminating pod ${RUNPOD_POD_ID}..."
    curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"${RUNPOD_POD_ID}\\\" }) }\"}"
fi
