"""Test whether the model's predictions actually respond to different actions.

Takes a single context (L frames), runs inference with each of the 10 actions,
and checks if predictions differ meaningfully between actions.

Usage:
    python src/action_test.py --checkpoint experiments/run/best.pt --data data/episodes
"""

import argparse
from pathlib import Path
from itertools import combinations

import torch
import numpy as np
from PIL import Image

from model import make_denoiser
from episode import Episode

# Action names matching ACTIONS_V2 in collect.py
ACTION_NAMES = [
    "forward", "backward", "strafe_L", "strafe_R",
    "turn_L", "turn_R", "attack", "atk+fwd",
    "fwd+turnL", "fwd+turnR",
]


def load_model(checkpoint_path, device, **kwargs):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_denoiser(**kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']+1}, loss={ckpt['loss']:.4f}")
    return model


def tensor_to_uint8(t):
    """Convert [-1,1] float tensor to uint8."""
    return ((t + 1) / 2 * 255).clamp(0, 255).byte()


def run_action_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(
        args.checkpoint, device,
        num_actions=args.num_actions,
        img_size=args.res,
        num_context_frames=args.num_context,
        model_size=args.model_size,
    )

    # Load a seed episode
    episode_dir = Path(args.data)
    episode_files = sorted(episode_dir.glob("episode_*.pt"))
    if not episode_files:
        print(f"No episodes found in {args.data}")
        return

    ep = Episode.load(episode_files[args.episode_idx])
    L = args.num_context
    start_t = L + 10  # skip first few frames
    context = ep.obs[start_t - L : start_t].unsqueeze(0).to(device)  # (1, L, C, H, W)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Generate prediction for each action ──
    print("\n=== Generating predictions for all 10 actions ===")
    predictions = {}  # action_idx -> (C, H, W) tensor in [-1,1]

    for act_idx in range(args.num_actions):
        action = torch.tensor([act_idx], device=device)
        with torch.no_grad():
            pred = model.sample(context, action, num_steps=args.num_denoise_steps)
        predictions[act_idx] = pred[0].cpu()

    # Save grid: context frame + 10 action predictions
    ctx_img = tensor_to_uint8(context[0, -1].cpu())  # last context frame
    pred_imgs = [tensor_to_uint8(predictions[i]) for i in range(args.num_actions)]

    # Build grid image: 1 row of context + 10 predictions
    all_imgs = [ctx_img] + pred_imgs
    h, w = all_imgs[0].shape[1], all_imgs[0].shape[2]

    # Arrange in 2 rows: top row = ctx + actions 0-4, bottom = actions 5-9
    top_row = torch.cat([all_imgs[0]] + all_imgs[1:6], dim=2)  # ctx + 5 actions
    # Pad first slot of bottom row with black
    bottom_row = torch.cat([torch.zeros_like(all_imgs[0])] + all_imgs[6:], dim=2)
    grid = torch.cat([top_row, bottom_row], dim=1)

    grid_img = Image.fromarray(grid.permute(1, 2, 0).numpy())
    grid_img.save(out_dir / "action_grid.png")
    print(f"Saved action grid to {out_dir / 'action_grid.png'}")

    # Also save individual labeled predictions
    for i in range(args.num_actions):
        img = Image.fromarray(pred_imgs[i].permute(1, 2, 0).numpy())
        img.save(out_dir / f"action_{i}_{ACTION_NAMES[i]}.png")

    # ── Part 2: Pairwise MSE between different actions ──
    print("\n=== Pairwise MSE between action predictions ===")
    mse_matrix = np.zeros((args.num_actions, args.num_actions))
    for i, j in combinations(range(args.num_actions), 2):
        mse = ((predictions[i].float() - predictions[j].float()) ** 2).mean().item()
        mse_matrix[i][j] = mse
        mse_matrix[j][i] = mse

    # Print MSE matrix
    header = "          " + "".join(f"{ACTION_NAMES[i]:>10s}" for i in range(args.num_actions))
    print(header)
    for i in range(args.num_actions):
        row = f"{ACTION_NAMES[i]:>10s}" + "".join(f"{mse_matrix[i][j]:10.5f}" for j in range(args.num_actions))
        print(row)

    cross_action_mse = np.mean([mse_matrix[i][j] for i, j in combinations(range(args.num_actions), 2)])
    print(f"\nMean cross-action MSE: {cross_action_mse:.6f}")

    # ── Part 3: Same action, different noise seeds ──
    print("\n=== Same action, different noise seeds (baseline variance) ===")
    seed_mses = []
    test_action = 0  # test with "forward"
    seed_preds = []
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        action = torch.tensor([test_action], device=device)
        with torch.no_grad():
            pred = model.sample(context, action, num_steps=args.num_denoise_steps)
        seed_preds.append(pred[0].cpu())

    for i, j in combinations(range(len(seed_preds)), 2):
        mse = ((seed_preds[i].float() - seed_preds[j].float()) ** 2).mean().item()
        seed_mses.append(mse)

    same_action_mse = np.mean(seed_mses)
    print(f"Mean same-action MSE (different seeds): {same_action_mse:.6f}")

    # ── Summary ──
    ratio = cross_action_mse / same_action_mse if same_action_mse > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"Cross-action MSE:  {cross_action_mse:.6f}")
    print(f"Same-action MSE:   {same_action_mse:.6f}")
    print(f"Ratio (higher=better): {ratio:.2f}x")
    print(f"{'='*60}")

    if ratio > 2.0:
        print("PASS: Model clearly responds to different actions")
    elif ratio > 1.2:
        print("WEAK: Model shows some action sensitivity, but not strong")
    else:
        print("FAIL: Model appears to ignore actions (predictions nearly identical)")

    # Save summary
    with open(out_dir / "action_test_results.txt", "w") as f:
        f.write(f"Cross-action MSE: {cross_action_mse:.6f}\n")
        f.write(f"Same-action MSE:  {same_action_mse:.6f}\n")
        f.write(f"Ratio: {ratio:.2f}x\n")
        f.write(f"Verdict: {'PASS' if ratio > 2.0 else 'WEAK' if ratio > 1.2 else 'FAIL'}\n")

    print(f"\nResults saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test action conditioning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/episodes")
    parser.add_argument("--output", type=str, default="experiments/action_test")
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--res", type=int, default=84)
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--num_denoise_steps", type=int, default=3)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of random seeds for same-action variance")
    args = parser.parse_args()
    run_action_test(args)


if __name__ == "__main__":
    main()
