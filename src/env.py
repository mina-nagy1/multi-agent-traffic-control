import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces


class TrafficEnv(gym.Env):
    """
    Single-intersection adaptive traffic signal control.

    Observation:
        Normalised queue lengths for 8 incoming lanes, shape=(8,), range=[0,1].

    Action:
        Discrete(2): 0 = NS green, 1 = EW green.

    Reward:
        -avg_waiting_time / 50  (negative, agent maximises by minimising wait).

    Episode:
        3600 simulation steps (~1 simulated hour).
        Each agent decision advances the sim by DELTA seconds.
    """

    DELTA     = 5       # simulation seconds per agent decision
    MIN_GREEN = 10      # minimum seconds before a phase switch is allowed
    YELLOW    = 4       # yellow-phase duration (seconds)
    MAX_Q     = 30.0    # normalisation constant (vehicles per lane)
    EP_LEN    = 3600    # episode length in simulation steps

    LANES = [
        "n2c_0", "n2c_1",
        "s2c_0", "s2c_1",
        "e2c_0", "e2c_1",
        "w2c_0", "w2c_1",
    ]

    def __init__(self, cfg, gui=False):
        """
        Args:
            cfg: Path to the .sumocfg file.
            gui: If True, launches sumo-gui instead of sumo.
        """
        super().__init__()
        self.cfg    = cfg
        self.binary = "sumo-gui" if gui else "sumo"
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(8,), dtype=np.float32)
        self._reset_state()

    def _reset_state(self):
        self._tls    = None
        self._gp     = []
        self._g2y    = {}
        self._cur    = 0
        self._gtimer = 0
        self._step   = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except Exception:
            pass
        traci.start([self.binary, "-c", self.cfg,
                     "--no-step-log", "true", "--no-warnings", "true"])
        self._reset_state()
        self._tls = traci.trafficlight.getIDList()[0]
        self._discover_phases()
        traci.trafficlight.setPhase(self._tls, self._gp[0])
        for _ in range(30):
            traci.simulationStep()
        self._step = 30
        return self._obs(), {}

    def _discover_phases(self):
        logic  = traci.trafficlight.getCompleteRedYellowGreenDefinition(
            self._tls)[0]
        phases = logic.phases
        self._gp = [i for i, p in enumerate(phases)
                    if "G" in p.state and "y" not in p.state.lower()]
        for g in self._gp:
            nxt = g + 1
            if nxt < len(phases) and "y" in phases[nxt].state.lower():
                self._g2y[g] = nxt
            else:
                for i, p in enumerate(phases):
                    if "y" in p.state.lower():
                        self._g2y[g] = i
                        break

    def step(self, action):
        action  = int(action)
        desired = self._gp[action]
        current = self._gp[self._cur]
        if desired != current and self._gtimer >= self.MIN_GREEN:
            traci.trafficlight.setPhase(self._tls, self._g2y[current])
            for _ in range(self.YELLOW):
                traci.simulationStep()
                self._step += 1
            self._cur    = action
            self._gtimer = 0
            traci.trafficlight.setPhase(self._tls, self._gp[self._cur])

        total_wait = 0.0
        for _ in range(self.DELTA):
            traci.simulationStep()
            self._step   += 1
            total_wait   += self._sum_wait()
        self._gtimer += self.DELTA

        avg_wait   = total_wait / self.DELTA
        reward     = -avg_wait / 50.0
        terminated = self._step >= self.EP_LEN
        if terminated:
            try:
                traci.close()
            except Exception:
                pass
        return (self._obs(), reward, terminated, False,
                {"avg_wait": avg_wait, "step": self._step})

    def _obs(self):
        q = np.array([self._q(l) for l in self.LANES], dtype=np.float32)
        return np.clip(q / self.MAX_Q, 0.0, 1.0)

    def _q(self, lane):
        try:
            return float(traci.lane.getLastStepHaltingNumber(lane))
        except Exception:
            return 0.0

    def _sum_wait(self):
        try:
            return sum(traci.vehicle.getWaitingTime(v)
                       for v in traci.vehicle.getIDList())
        except Exception:
            return 0.0

    def close(self):
        try:
            traci.close()
        except Exception:
            pass


class NoisyTrafficEnv(TrafficEnv):
    """
    TrafficEnv with configurable sensor noise.

    Models two types of real-world detection imperfection:

    Gaussian noise (sigma):
        Random measurement error added to every lane count.
        Simulates varying detection confidence, lighting changes, and
        YOLO bounding-box uncertainty.

    Dropout / occlusion (p_drop):
        Randomly zeros out entire lane readings.
        Simulates parked vehicles blocking camera view, lens obstruction,
        or complete sensor failure on a lane.

    In a real deployment (CARLA + YOLO pipeline), both errors occur
    simultaneously. This parametric model lets us study the effect of
    each independently and in combination.

    Args:
        sigma:  Gaussian noise std as a fraction of MAX_Q. 0 = clean.
        p_drop: Probability that any lane reading is lost each step. 0 = clean.
        seed:   RNG seed for reproducibility.
    """

    def __init__(self, cfg, sigma=0.0, p_drop=0.0, seed=42, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.sigma  = sigma
        self.p_drop = p_drop
        self._rng   = np.random.default_rng(seed)

    def _obs(self):
        true_q = np.array([self._q(l) for l in self.LANES], dtype=np.float32)
        obs    = np.clip(true_q / self.MAX_Q, 0.0, 1.0)

        if self.sigma > 0:
            noise = self._rng.normal(0.0, self.sigma,
                                     size=obs.shape).astype(np.float32)
            obs = obs + noise

        if self.p_drop > 0:
            mask = self._rng.random(size=obs.shape) < self.p_drop
            obs[mask] = 0.0

        return np.clip(obs, 0.0, 1.0).astype(np.float32)
