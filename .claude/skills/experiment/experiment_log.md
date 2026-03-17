# Experiment Log

Operational log of training runs. Updated after each /experiment invocation.

## Run 9 (2026-03-17): 160x120, large model, DIAMOND recipe
- **Config**: large 20.6M, 160x120 native, 5K eps, batch=16, noise_aug=0.3, AdamW+warmup
- **Training**: 1 epoch (~66K steps), loss=0.061
- **Eval**: TF PSNR=19.3 dB, AR=17.4 dB, copy baseline=19.5 dB
- **Verdict**: Undertrained. 160x120 needs more epochs. Loss barely converged (0.061 vs 84x84's 0.027)
- **Lesson**: At higher res, same model needs proportionally more training time. 1 epoch not enough.

## Run 7 (2026-03-16): 84x84, large model, 10K episodes
- **Config**: large 20.6M, 84x84, 10K eps, batch=64, noise_aug=0.1, RAM dataset
- **Training**: 3 epochs, loss=0.027
- **Eval**: TF PSNR=21.4 dB, AR=19.2 dB, copy baseline=21.0 dB
- **Verdict**: Best 84x84 result. TF beats copy by +0.33 dB. Drift still kills AR quality.
- **Lesson**: Teacher-forced is good, drift is the bottleneck, not model quality.

## Key metrics to track
- Teacher-forced PSNR (does model learn at all?)
- Delta vs copy baseline (is model better than doing nothing?)
- PSNR at step 1 vs step 32 (drift rate)
- Action ratio (controllability)
