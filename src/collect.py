"""Collect frame+action episodes from ViZDoom for world model training.

Usage:
    python src/collect.py --num_episodes 100 --output data/episodes
    python src/collect.py --num_episodes 1000 --output data/episodes --scenario deathmatch
"""

import argparse
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image

from episode import Episode


# Discrete action space: 7 actions for FPS gameplay
# Each action is a list of button states
ACTIONS = [
    [0, 0, 0, 0, 0, 1, 0],  # 0: move forward
    [0, 0, 0, 0, 0, 0, 1],  # 1: move backward
    [0, 0, 0, 1, 0, 0, 0],  # 2: move left
    [0, 0, 0, 0, 1, 0, 0],  # 3: move right
    [0, 0, 1, 0, 0, 0, 0],  # 4: turn left
    [0, 0, 0, 0, 0, 0, 0],  # 5: turn right (placeholder, see BUTTONS)
    [1, 0, 0, 0, 0, 0, 0],  # 6: attack
]

BUTTONS = [
    vzd.Button.ATTACK,
    vzd.Button.SPEED,
    vzd.Button.TURN_LEFT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
]

# Fix action 5 (turn right): no TURN_RIGHT button, we use TURN_LEFT_RIGHT_DELTA instead
# Actually simpler: just use TURN_LEFT and TURN_RIGHT as separate buttons
BUTTONS_V2 = [
    vzd.Button.ATTACK,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
]

ACTIONS_V2 = [
    [0, 1, 0, 0, 0, 0, 0],  # 0: move forward
    [0, 0, 1, 0, 0, 0, 0],  # 1: move backward
    [0, 0, 0, 1, 0, 0, 0],  # 2: strafe left
    [0, 0, 0, 0, 1, 0, 0],  # 3: strafe right
    [0, 0, 0, 0, 0, 1, 0],  # 4: turn left
    [0, 0, 0, 0, 0, 0, 1],  # 5: turn right
    [1, 0, 0, 0, 0, 0, 0],  # 6: attack
    [1, 1, 0, 0, 0, 0, 0],  # 7: attack + forward
    [0, 1, 0, 0, 0, 1, 0],  # 8: forward + turn left
    [0, 1, 0, 0, 0, 0, 1],  # 9: forward + turn right
]

NUM_ACTIONS = len(ACTIONS_V2)


class ScriptedPolicy:
    """Simple scripted policy: mostly forward, occasional turns, shoot periodically.

    Produces coherent movement sequences instead of random spinning.
    """

    def __init__(self, forward_prob=0.45, turn_prob=0.15, shoot_prob=0.10):
        self.forward_prob = forward_prob
        self.turn_prob = turn_prob
        self.shoot_prob = shoot_prob
        self._turn_dir = None  # current turn direction (persist for smoother turns)
        self._turn_steps = 0

    def __call__(self):
        r = np.random.random()

        # If mid-turn, continue for a few steps (smoother rotation)
        if self._turn_steps > 0:
            self._turn_steps -= 1
            # 8: fwd+turnL, 9: fwd+turnR (move while turning)
            return 8 if self._turn_dir == "left" else 9

        # Start a new turn sequence
        if r < self.turn_prob:
            self._turn_dir = np.random.choice(["left", "right"])
            self._turn_steps = np.random.randint(2, 6)  # turn for 2-6 steps
            return 8 if self._turn_dir == "left" else 9

        # Move forward (most common)
        if r < self.turn_prob + self.forward_prob:
            return 0  # forward

        # Shoot while moving forward
        if r < self.turn_prob + self.forward_prob + self.shoot_prob:
            return 7  # attack + forward

        # Occasional strafe
        if r < self.turn_prob + self.forward_prob + self.shoot_prob + 0.10:
            return np.random.choice([2, 3])  # strafe left/right

        # Fallback: random action for diversity
        return np.random.randint(NUM_ACTIONS)


SCREEN_RESOLUTIONS = {
    "160x120": vzd.ScreenResolution.RES_160X120,
    "320x240": vzd.ScreenResolution.RES_320X240,
    "256x144": vzd.ScreenResolution.RES_256X144,
    "256x160": vzd.ScreenResolution.RES_256X160,
    "640x480": vzd.ScreenResolution.RES_640X480,
}


def make_game(scenario: str = "deathmatch", res: int = 84, frame_skip: int = 4,
              screen_res: str = "160x120") -> vzd.DoomGame:
    game = vzd.DoomGame()

    # Scenario
    scenario_path = vzd.scenarios_path + f"/{scenario}.cfg"
    game.load_config(scenario_path)

    # Override screen settings
    vzd_res = SCREEN_RESOLUTIONS.get(screen_res, vzd.ScreenResolution.RES_160X120)
    game.set_screen_resolution(vzd_res)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_render_hud(False)

    # Clear default buttons and set ours
    game.clear_available_buttons()
    for btn in BUTTONS_V2:
        game.add_available_button(btn)

    # Performance
    game.set_ticrate(35)  # DOOM default

    game.init()
    return game


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, C) frame. size=0 means keep native resolution."""
    if size == 0:
        return frame
    img = Image.fromarray(frame)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def collect_episode(game: vzd.DoomGame, res: int = 84, frame_skip: int = 4,
                    max_steps: int = 1000, policy: str = "random") -> Episode:
    """Collect a single episode."""
    game.new_episode()

    obs_list = []
    act_list = []
    rew_list = []
    end_list = []
    trunc_list = []

    # Initial observation
    state = game.get_state()
    frame = resize_frame(state.screen_buffer, res)
    obs_list.append(frame)

    steps = 0
    # Create policy
    if policy == "random":
        policy_fn = lambda: np.random.randint(NUM_ACTIONS)
    elif policy == "scripted":
        policy_fn = ScriptedPolicy()
    elif policy == "mixed":
        scripted = ScriptedPolicy()
        # 70% scripted, 30% random
        def policy_fn():
            if np.random.random() < 0.7:
                return scripted()
            return np.random.randint(NUM_ACTIONS)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    while not game.is_episode_finished() and steps < max_steps:
        # Select action
        action_idx = policy_fn()

        action = ACTIONS_V2[action_idx]

        # Step (with frame skip)
        reward = game.make_action(action, frame_skip)

        # Record action and reward
        act_list.append(action_idx)
        rew_list.append(reward)

        if game.is_episode_finished():
            # Terminal: create a black frame as final obs (match shape of previous frame)
            end_list.append(True)
            trunc_list.append(False)
            obs_list.append(np.zeros_like(obs_list[-1]))
        else:
            end_list.append(False)
            trunc_list.append(steps + 1 >= max_steps)
            state = game.get_state()
            frame = resize_frame(state.screen_buffer, res)
            obs_list.append(frame)

        steps += 1

    return Episode.from_numpy(obs_list, act_list, rew_list, end_list, trunc_list)


def main():
    parser = argparse.ArgumentParser(description="Collect ViZDoom episodes")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/episodes")
    parser.add_argument("--scenario", type=str, default="deathmatch")
    parser.add_argument("--res", type=int, default=0,
                        help="Resize to square NxN (0=native resolution)")
    parser.add_argument("--screen_res", type=str, default="160x120",
                        choices=list(SCREEN_RESOLUTIONS.keys()),
                        help="ViZDoom render resolution")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode (after frame skip)")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "scripted", "mixed"],
                        help="random: uniform, scripted: forward+turns, mixed: 70%% scripted 30%% random")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.num_episodes} episodes from {args.scenario}")
    print(f"Resolution: {args.res}x{args.res}, frame_skip: {args.frame_skip}")
    print(f"Output: {output_dir}")

    game = make_game(args.scenario, args.res, args.frame_skip, args.screen_res)

    total_frames = 0
    t0 = time.time()

    for i in range(args.num_episodes):
        ep = collect_episode(game, args.res, args.frame_skip, args.max_steps, args.policy)
        ep_path = output_dir / f"episode_{i:06d}.pt"
        ep.save(ep_path)

        total_frames += len(ep)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            fps = total_frames / elapsed
            print(f"  Episode {i+1}/{args.num_episodes}: "
                  f"{len(ep)} steps, total {total_frames} frames, "
                  f"{fps:.0f} fps, {elapsed:.1f}s elapsed")

    game.close()

    elapsed = time.time() - t0
    print(f"\nDone! {args.num_episodes} episodes, {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.0f} fps)")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
