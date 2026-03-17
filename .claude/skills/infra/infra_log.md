# Infrastructure Log

Operational log of pod management, data transfers, and cost tracking.

## Active Resources
- **Pods**: All stopped/terminated as of 2026-03-17
- **W&B artifacts**: quake3-wm-run7-large-10k-ep3, quake3-wm-160x120-large
- **HF datasets**: rzhang139/vizdoom-episodes (episodes_160x120_5k, episodes_84x84_10k)

## Pod History
| Pod ID | GPU | Cloud | Hours | Cost | Purpose | Status |
|--------|-----|-------|-------|------|---------|--------|
| noecq0fv7nifkx | 3090 | Community | ~30h | ~$6.60 | Runs 2-4 | Terminated |
| fuhugooncxvd4d | 3090 | Community | ~12h | ~$2.64 | Run 5 (5K eps) | Terminated |
| wor8pz6pp15sds | 3090 | Secure | ~14h | ~$5.04 | Run 7 (10K RAM) | Stopped |
| 8ckjmv5d73pvov | 3090 | Secure | ~12h | ~$4.32 | Run 9 (160x120) | Stopped |
| odmf8piijv526s | 3090 | Community | ~2h | ~$0.44 | DIAMOND fork (failed) | Terminated |

## Gotchas Discovered
- Community pods get preempted overnight — always use SECURE for >2h runs
- Streaming dataset = 0% GPU — always use RAM mode
- RunPod API gpuUtilPercent unreliable — verify with nvidia-smi
- 10K episodes at 84x84 = ~40GB, needs 62GB+ RAM pod for RAM mode
- Pod with <48GB RAM + 10K episodes OOMs during data loading
- timed auto-shutdown (sleep N) more reliable than poll-based shutdown

## Total spend: ~$25
