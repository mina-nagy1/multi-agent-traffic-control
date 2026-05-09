import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict


class GridTrafficEnv(gym.Env):
    """
    Multi-agent traffic signal control on a 3x3 grid (9 intersections).

    Each intersection is treated as an independent agent.

    Per-agent observation (shape 8):
        Normalised queue lengths on the 8 incoming lanes (2 per direction).

    Per-agent action:
        Discrete(2): 0 = NS green, 1 = EW green.

    Reward:
        Each agent receives its local reward: -avg_queue / 50.

    Global state (shape 72):
        Concatenation of all 9 agents' observations.
        Used by the MAPPO centralised critic during training only.

    Args:
        scenario:  Traffic demand scenario.
                   One of "uniform", "heavy_ew", "rush_hour".
        net_dir:   Directory containing the SUMO grid network files.
        gui:       Launch sumo-gui if True.
    """

    GRID      = 3
    N_AGENTS  = 9
    OBS_DIM   = 8
    DELTA     = 5
    MIN_GREEN = 10
    YELLOW    = 4
    MAX_Q     = 30.0
    EP_LEN    = 3600

    def __init__(self, scenario="uniform",
                 net_dir="/content/sumo_grid", gui=False):
        super().__init__()
        self.cfg    = f"{net_dir}/grid_{scenario}.sumocfg"
        self.binary = "sumo-gui" if gui else "sumo"

        self.observation_space = spaces.Box(
            0.0, 1.0,
            shape=(self.N_AGENTS, self.OBS_DIM),
            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * self.N_AGENTS)

        self.agent_ids = [f"I{r}{c}"
                          for r in range(self.GRID)
                          for c in range(self.GRID)]
        self._lanes     = self._build_lane_map()
        self._tls_map   = {}
        self._gp_map    = {}
        self._g2y_map   = {}
        self._cur_map   = {}
        self._timer_map = {}
        self._step      = 0

    def _build_lane_map(self):
        lanes = {}
        G = self.GRID
        for r in range(G):
            for c in range(G):
                iid      = f"I{r}{c}"
                incoming = []
                # North arm
                if r == 0:
                    incoming += [f"nIn{c}_0", f"nIn{c}_1"]
                else:
                    incoming += [f"v{r-1}{c}d_0", f"v{r-1}{c}d_1"]
                # South arm
                if r == G - 1:
                    incoming += [f"sIn{c}_0", f"sIn{c}_1"]
                else:
                    incoming += [f"v{r}{c}u_0", f"v{r}{c}u_1"]
                # West arm
                if c == 0:
                    incoming += [f"wIn{r}_0", f"wIn{r}_1"]
                else:
                    incoming += [f"h{r}{c-1}r_0", f"h{r}{c-1}r_1"]
                # East arm
                if c == G - 1:
                    incoming += [f"eIn{r}_0", f"eIn{r}_1"]
                else:
                    incoming += [f"h{r}{c}l_0", f"h{r}{c}l_1"]
                lanes[iid] = incoming
        return lanes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except Exception:
            pass
        traci.start([self.binary, "-c", self.cfg,
                     "--no-step-log", "true", "--no-warnings", "true"])
        all_tls = traci.trafficlight.getIDList()
        for i, aid in enumerate(self.agent_ids):
            tls_id = all_tls[i] if i < len(all_tls) else all_tls[0]
            for t in all_tls:
                if aid in t or t in aid:
                    tls_id = t
                    break
            self._tls_map[aid]   = tls_id
            self._gp_map[aid]    = self._get_green_phases(tls_id)
            self._g2y_map[aid]   = self._get_g2y(tls_id)
            self._cur_map[aid]   = 0
            self._timer_map[aid] = 0
            traci.trafficlight.setPhase(tls_id, self._gp_map[aid][0])
        self._step = 0
        for _ in range(30):
            traci.simulationStep()
        self._step = 30
        return self._obs_all(), {}

    def _get_green_phases(self, tls_id):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        gp    = [i for i, p in enumerate(logic.phases)
                 if "G" in p.state and "y" not in p.state.lower()]
        return gp if gp else [0]

    def _get_g2y(self, tls_id):
        logic  = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        phases = logic.phases
        gp     = self._get_green_phases(tls_id)
        g2y    = {}
        for g in gp:
            nxt = g + 1
            if nxt < len(phases) and "y" in phases[nxt].state.lower():
                g2y[g] = nxt
            else:
                for i, p in enumerate(phases):
                    if "y" in p.state.lower():
                        g2y[g] = i
                        break
        return g2y

    def step(self, actions):
        for i, aid in enumerate(self.agent_ids):
            action  = int(actions[i])
            gp      = self._gp_map[aid]
            desired = gp[action]
            current = gp[self._cur_map[aid]]
            tls     = self._tls_map[aid]
            if desired != current and self._timer_map[aid] >= self.MIN_GREEN:
                traci.trafficlight.setPhase(tls, self._g2y_map[aid][current])
                for _ in range(self.YELLOW):
                    traci.simulationStep()
                    self._step += 1
                self._cur_map[aid]   = action
                self._timer_map[aid] = 0
                traci.trafficlight.setPhase(tls, gp[action])

        total_wait = defaultdict(float)
        for _ in range(self.DELTA):
            traci.simulationStep()
            self._step += 1
            for aid in self.agent_ids:
                for lane in self._lanes[aid]:
                    try:
                        total_wait[aid] += float(
                            traci.lane.getLastStepHaltingNumber(lane))
                    except Exception:
                        pass

        for aid in self.agent_ids:
            self._timer_map[aid] += self.DELTA

        avg_waits   = {aid: total_wait[aid] / self.DELTA
                       for aid in self.agent_ids}
        rewards     = np.array([-avg_waits[aid] / 50.0
                                 for aid in self.agent_ids],
                                dtype=np.float32)
        global_wait = float(np.mean(list(avg_waits.values())))
        terminated  = self._step >= self.EP_LEN
        if terminated:
            try:
                traci.close()
            except Exception:
                pass
        obs  = self._obs_all()
        info = {
            "avg_wait":       global_wait,
            "per_agent_wait": avg_waits,
            "global_state":   obs.flatten(),
            "step":           self._step,
        }
        return obs, rewards, terminated, False, info

    def _obs_all(self):
        obs = np.zeros((self.N_AGENTS, self.OBS_DIM), dtype=np.float32)
        for i, aid in enumerate(self.agent_ids):
            for j, lane in enumerate(self._lanes[aid]):
                try:
                    obs[i, j] = min(
                        traci.lane.getLastStepHaltingNumber(lane) / self.MAX_Q,
                        1.0)
                except Exception:
                    pass
        return obs

    def global_state(self):
        """Returns flattened (72,) global observation for MAPPO critic."""
        return self._obs_all().flatten()

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
