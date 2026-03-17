---
name: infra
description: Manage RunPod pods, W&B artifacts, HuggingFace datasets. Use for pod lifecycle, data transfers, cost tracking.
argument-hint: "[action: create|stop|status|upload|download]"
user-invocable: true
hooks:
  Stop:
    - hooks:
        - type: prompt
          prompt: |
            Infra exit checklist (skip items that don't apply):
            1. No pods left running unintentionally?
            2. Checkpoints uploaded to W&B?
            3. Datasets uploaded to HuggingFace?
            4. Cost estimate logged to infra_log.md?
            Return {"ok": true} — this is advisory.
---

# Infrastructure Skill

Manage GPU pods, artifact storage, and dataset syncing.

## API Keys
Read from `../personal-research/runpod/.env`:
- `RUNPOD_API_KEY` — pod management
- `WANDB_API_KEY` — artifact upload
- `HF_TOKEN` — dataset upload
- `GITHUB_TOKEN` — repo access

## RunPod Pod Management

### Create pod
```bash
curl -s --request POST \
  --header 'content-type: application/json' \
  --url "https://api.runpod.io/graphql?api_key=<KEY>" \
  --data '{"query":"mutation { podFindAndDeployOnDemand(input: { name: \"<NAME>\", gpuTypeId: \"NVIDIA GeForce RTX 3090\", gpuCount: 1, volumeInGb: 100, containerDiskInGb: 50, templateId: \"runpod-torch-v240\", cloudType: SECURE, minMemoryInGb: 62 }) { id desiredStatus machine { podHostId } } }"}'
```

### Check GPU utilization
```bash
curl -s ... '{"query":"query { pod(input: {podId: \"<ID>\"}) { runtime { gpus { gpuUtilPercent memoryUtilPercent } } } }"}'
```

### Stop pod
```bash
curl -s ... '{"query":"mutation { podStop(input: {podId: \"<ID>\"}) { id desiredStatus } }"}'
```

## Pod specs (proven configs)
- **Large model (20.6M) batch=64**: RTX 3090, 62GB+ RAM, 100GB disk, SECURE cloud
- **Large model 160x120 batch=16**: same specs
- **Eval only**: any RTX 3090, 20GB disk, community OK

## SSH via RunPod gateway
```
Host runpod-new
  HostName ssh.runpod.io
  User <POD_ID>-<HOST_ID>
  Port 22
  IdentityFile ~/.ssh/runpod
```
Always use heredocs. RunPod SSH ignores command args.

## W&B uploads
Artifact naming: `quake3-wm-<description>`
ALWAYS upload before stopping pods.

## HuggingFace datasets
Repo: `rzhang139/vizdoom-episodes`
Download on pod:
```bash
huggingface-cli download rzhang139/vizdoom-episodes --repo-type dataset --include "episodes_160x120_5k/*" --local-dir data --token $HF_TOKEN
```

## Auto-shutdown pattern
For overnight runs:
```bash
nohup bash -c 'sleep 18000 && pkill -f train.py && sleep 10 && <upload> && <stop pod>' > /workspace/shutdown.log 2>&1 &
```

## Cost tracking
Always estimate: hours * $/hr (RTX 3090 SECURE ≈ $0.36/hr, community ≈ $0.22/hr)

## Gotchas
- Community/spot pods get preempted — use SECURE for overnight
- `wandb/` dir in repo shadows the wandb package — run W&B from /tmp or outside repo
- num_workers=0 + streaming dataset = 0% GPU util — always use RAM mode
- RunPod API `gpuUtilPercent` is sometimes wrong — verify with `nvidia-smi` via SSH
- scp doesn't work through RunPod SSH gateway — use git, W&B artifacts, or HF

## Arguments
$ARGUMENTS
