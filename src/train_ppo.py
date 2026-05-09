"""
Train a PPO agent on the single-intersection environment.

Usage:
    python src/train_ppo.py --cfg /content/sumo_net/net.sumocfg \
                            --steps 50000 \
                            --save_dir ./checkpoints
"""
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import TrafficEnv, NoisyTrafficEnv


class WaitTimeCallback(BaseCallback):
    """Records per-episode average waiting time during training."""

    def __init__(self):
        super().__init__()
        self.episode_waits = []
        self._buf = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "avg_wait" in info:
            self._buf.append(info["avg_wait"])
        if self.locals.get("dones", [False])[0] and self._buf:
            self.episode_waits.append(np.mean(self._buf))
            self._buf = []
        return True


def train(cfg, total_steps=50_000, save_dir="./checkpoints",
          sigma=0.0, p_drop=0.0, learning_rate=3e-4):
    """
    Train a PPO agent and save the final model.

    Args:
        cfg:           Path to .sumocfg file.
        total_steps:   Total training timesteps.
        save_dir:      Directory for checkpoints.
        sigma:         Gaussian noise level (0 = clean observations).
        p_drop:        Lane dropout probability (0 = no occlusion).
        learning_rate: Adam learning rate.

    Returns:
        Trained SB3 PPO model.
    """
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        if sigma > 0 or p_drop > 0:
            return Monitor(NoisyTrafficEnv(cfg=cfg, sigma=sigma, p_drop=p_drop))
        return Monitor(TrafficEnv(cfg=cfg))

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate = learning_rate,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
        verbose       = 1,
    )

    wait_cb = WaitTimeCallback()
    ckpt_cb = CheckpointCallback(
        save_freq   = 5_000,
        save_path   = save_dir,
        name_prefix = "ppo",
    )

    model.learn(
        total_timesteps = total_steps,
        callback        = [ckpt_cb, wait_cb],
        progress_bar    = True,
    )

    final_path = os.path.join(save_dir, "ppo_final")
    model.save(final_path)
    print(f"Model saved to {final_path}.zip")
    vec_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",      required=True,
                        help="Path to .sumocfg file")
    parser.add_argument("--steps",    type=int, default=50_000,
                        help="Total training timesteps")
    parser.add_argument("--save_dir", default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--sigma",    type=float, default=0.0,
                        help="Gaussian noise sigma")
    parser.add_argument("--p_drop",   type=float, default=0.0,
                        help="Lane dropout probability")
    parser.add_argument("--lr",       type=float, default=3e-4,
                        help="Learning rate")
    args = parser.parse_args()

    train(
        cfg          = args.cfg,
        total_steps  = args.steps,
        save_dir     = args.save_dir,
        sigma        = args.sigma,
        p_drop       = args.p_drop,
        learning_rate= args.lr,
    )
