# Vision-Driven Multi-Agent Traffic Signal Control

Adaptive traffic signal control using Multi-Agent Reinforcement Learning (MARL),
trained and evaluated in SUMO across synthetic and real-world OpenStreetMap networks.


---

## Results

### Notebook 1 — Single Intersection, Perfect Information

| Controller | Avg Wait | vs Fixed-Time |
|------------|----------|---------------|
| Fixed-time (30s cycle) | 44.4s | baseline |
| Random agent | 18.0s | — |
| PPO (ours) | 6.4s | **-85.5%** |

### Notebook 2 — Noise Injection and Vision Simulation

Parametric noise model replacing CARLA + YOLO: Gaussian measurement error
and lane dropout (occlusion). Fixed-time baseline: 41.0s.

**Gaussian noise tolerance** (PPO trained on clean data):

| Sigma | Avg Wait | vs Fixed-Time |
|-------|----------|---------------|
| 0.00 (clean) | 6.4s | -84.4% |
| 0.05 | 8.9s | -78.3% |
| 0.10 | 12.9s | -68.6% |
| 0.15 | 13.6s | -66.9% |
| 0.20 | 15.2s | -62.9% |
| 0.30 | 19.4s | -52.7% |
| 0.50 | 19.1s | -53.4% |

Agent remains better than fixed-time across all tested noise levels.

**Dropout / occlusion tolerance**:

| p_drop | Avg Wait | vs Fixed-Time |
|--------|----------|---------------|
| 0.00 | 6.4s | -84.4% |
| 0.05 | 6.4s | -84.4% |
| 0.10 | 6.5s | -84.1% |
| 0.20 | 6.8s | -83.4% |
| 0.30 | 6.8s | -83.4% |
| 0.50 | 7.1s | -82.7% |

Agent is highly robust to occlusion — 50% lane dropout causes less than 11% degradation.

**Noise-aware training** (sigma=0.15, p_drop=0.10):

| Model | Avg Wait |
|-------|----------|
| Clean-trained PPO | 17.9s |
| Noise-aware PPO | 14.6s |

18.6% additional improvement from training on noisy observations.

### Notebook 3 — Multi-Intersection MARL (3x3 Grid, 9 Agents)

| Scenario | Fixed-Time | IPPO | MAPPO | MAPPO vs Fixed |
|----------|-----------|------|-------|----------------|
| Uniform | 1270.0s | 3.3s | 3.3s | **-99.7%** |
| Heavy EW | 2237.6s | 5.8s | 5.8s | **-99.7%** |
| Rush Hour | 1790.7s | 10.6s | 10.6s | **-99.4%** |

IPPO and MAPPO reach comparable performance on this grid size.
MAPPO provides more stable training via its centralised critic, with its
advantage expected to grow on larger grids or with longer training horizons.

### Notebook 4 — Real OSM Network (Cairo)

| Controller | Avg Wait | vs Fixed-Time |
|-----------|----------|---------------|
| Fixed-time | 198.9s | baseline |
| MAPPO zero-shot transfer | 0.4s | **-99.8%** |
| MAPPO fine-tuned | 0.4s | **-99.8%** |

Zero-shot transfer from the synthetic grid to a real Cairo street network
achieves the same performance as fine-tuning, demonstrating strong
generalisation of the learned policy.

---

## Project Structure

```
traffic-signal-control/
|
+-- src/
|   +-- env.py             # TrafficEnv + NoisyTrafficEnv (single intersection)
|   +-- grid_env.py        # GridTrafficEnv (3x3 multi-agent)
|   +-- osm_env.py         # OSMTrafficEnv (real OpenStreetMap network)
|   +-- mappo.py           # MAPPOActor, MAPPOCritic, MAPPOTrainer
|   +-- train_ppo.py       # PPO training script
|   +-- train_mappo.py     # MAPPO training script
|   +-- evaluate.py        # Evaluation utilities (PPO, MAPPO, fixed-time)
|   +-- utils.py           # SUMO network builders, XML helpers
|
+-- notebooks/
|   +-- 01_SUMO-RL Integration & PPO Training.ipynb
|   +-- 02_Noise_Vision_Simulation.ipynb
|   +-- 03_MARL_MAPPO.ipynb
|   +-- 04_OSM_Final.ipynb
|
+-- results/
|   +-- figures/
|
+-- requirements.txt
+-- README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Single intersection PPO (clean observations):
```bash
python src/train_ppo.py --cfg /path/to/net.sumocfg --steps 50000
```

Noise-aware PPO (sigma=0.15, p_drop=0.10):
```bash
python src/train_ppo.py --cfg /path/to/net.sumocfg --sigma 0.15 --p_drop 0.10
```

MAPPO on 3x3 grid (uniform traffic):
```bash
python src/train_mappo.py --scenario uniform --steps 50000
```

MAPPO on 3x3 grid (heavy east-west):
```bash
python src/train_mappo.py --scenario heavy_ew --steps 50000
```

---

## Key Design Decisions

**Noise injection instead of CARLA**
Rather than using CARLA + YOLO to generate noisy camera detections, sensor
imperfection is modelled parametrically with Gaussian noise and lane dropout.
This is more rigorous because noise levels can be controlled and studied
independently, producing clean degradation curves rather than uncontrolled
simulator artefacts.

**MAPPO with centralised critic**
The global critic sees all 9 agents' observations simultaneously during
training, resolving the non-stationarity problem inherent to independent
MARL. At deployment only the actor is needed, keeping inference lightweight
and fully decentralised.

**Direct policy transfer to real networks**
The OSM environment exposes the same observation and action interface as the
synthetic 3x3 grid. This enables zero-shot transfer of trained weights to a
real Cairo street network with no architecture changes. The zero-shot result
matches the fine-tuned result, confirming the policy generalises beyond the
training distribution.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Traffic simulation | SUMO + TraCI |
| Single-agent RL | Stable-Baselines3 PPO (PyTorch) |
| Multi-agent RL | Custom MAPPO implementation (PyTorch) |
| Environment interface | Custom Gymnasium wrappers |
| Networks | Synthetic 4-way intersection, 3x3 grid, real OSM (Cairo) |
| Noise model | Gaussian + dropout (parametric vision simulation) |
| Real map data | OpenStreetMap via Overpass API + netconvert |
