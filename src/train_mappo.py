"""
Train a MAPPO agent on the 3x3 grid or OSM environment.

Usage:
    python src/train_mappo.py --scenario uniform \
                              --net_dir /content/sumo_grid \
                              --steps 50000 \
                              --save_dir ./checkpoints
"""
import argparse
import os
import time
import numpy as np
import torch

from grid_env import GridTrafficEnv
from mappo import MAPPOTrainer


def train(scenario="uniform", net_dir="/content/sumo_grid",
          total_steps=50_000, rollout_steps=2048,
          save_dir="./checkpoints", lr=3e-4,
          resume_from=None):
    """
    Train MAPPO on the 3x3 grid environment.

    Args:
        scenario:      Traffic scenario. One of "uniform", "heavy_ew",
                       "rush_hour".
        net_dir:       Directory containing SUMO grid network files.
        total_steps:   Total training environment steps.
        rollout_steps: Steps collected per rollout before updating.
        save_dir:      Directory for checkpoints.
        lr:            Learning rate.
        resume_from:   Path to a .pt checkpoint to resume training from.

    Returns:
        Trained MAPPOTrainer instance.
    """
    os.makedirs(save_dir, exist_ok=True)

    if resume_from:
        trainer = MAPPOTrainer.load(resume_from, lr=lr)
        print(f"Resumed from {resume_from}")
    else:
        trainer = MAPPOTrainer(lr=lr)

    env    = GridTrafficEnv(scenario=scenario, net_dir=net_dir)
    obs, _ = env.reset()

    total        = 0
    rollout_n    = 0
    ep_waits     = []
    ep_buf       = []
    start_time   = time.time()

    print(f"Training MAPPO: scenario={scenario}, total_steps={total_steps}")

    while total < total_steps:
        for _ in range(rollout_steps):
            actions, log_probs = trainer.select_actions(obs)
            global_s           = env.global_state()
            value              = trainer.get_value(global_s)
            obs_next, rewards, done, trunc, info = env.step(actions)
            ep_buf.append(info["avg_wait"])
            trainer.store(obs, global_s, actions, rewards,
                          done or trunc, log_probs, value)
            obs    = obs_next
            total += 1
            if done or trunc:
                ep_waits.append(np.mean(ep_buf))
                ep_buf = []
                obs, _ = env.reset()

        last_val       = trainer.get_value(env.global_state())
        a_loss, c_loss = trainer.update(last_val)
        rollout_n     += 1
        elapsed        = time.time() - start_time
        ep_mean        = np.mean(ep_waits[-5:]) if ep_waits else 0.0

        print(f"  update {rollout_n:>3} | steps {total:>6} | "
              f"ep_wait {ep_mean:>7.2f}s | "
              f"actor {a_loss:>7.4f} | "
              f"critic {c_loss:>7.4f} | "
              f"fps {total / max(elapsed, 1):>4.0f}")

        if rollout_n % 5 == 0:
            ckpt_path = os.path.join(save_dir,
                                     f"mappo_step{total}.pt")
            trainer.save(ckpt_path, ep_waits)

    env.close()
    final_path = os.path.join(save_dir, "mappo_final.pt")
    trainer.save(final_path, ep_waits)
    print(f"Model saved to {final_path}")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",  default="uniform",
                        choices=["uniform", "heavy_ew", "rush_hour"])
    parser.add_argument("--net_dir",   default="/content/sumo_grid")
    parser.add_argument("--steps",     type=int, default=50_000)
    parser.add_argument("--rollout",   type=int, default=2048)
    parser.add_argument("--save_dir",  default="./checkpoints")
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--resume",    default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        scenario      = args.scenario,
        net_dir       = args.net_dir,
        total_steps   = args.steps,
        rollout_steps = args.rollout,
        save_dir      = args.save_dir,
        lr            = args.lr,
        resume_from   = args.resume,
    )
