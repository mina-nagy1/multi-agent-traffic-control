import numpy as np
import torch
import traci


def evaluate_ppo(model, env_class, env_kwargs, n_steps=500):
    """
    Evaluate a trained SB3 PPO model on a single-agent environment.

    Args:
        model:      Loaded SB3 PPO model.
        env_class:  Environment class (TrafficEnv or NoisyTrafficEnv).
        env_kwargs: Dict of keyword arguments passed to env_class.
        n_steps:    Number of agent decisions to evaluate.

    Returns:
        Tuple of (waits_list, mean_wait).
    """
    env = env_class(**env_kwargs)
    obs, _ = env.reset()
    waits  = []
    step   = 0
    while step < n_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(action)
        waits.append(info["avg_wait"])
        step += 1
        if done or trunc:
            break
    env.close()
    mean = float(np.mean(waits)) if waits else 0.0
    return waits, mean


def evaluate_mappo(actor, env, n_steps=500):
    """
    Evaluate a trained MAPPO actor on a multi-agent environment.

    Uses deterministic action selection (argmax of action probabilities).

    Args:
        actor:   MAPPOActor instance with loaded weights.
        env:     Instantiated multi-agent environment
                 (GridTrafficEnv or OSMTrafficEnv).
        n_steps: Number of environment steps to evaluate.

    Returns:
        Tuple of (waits_list, mean_wait).
    """
    actor.eval()
    obs, _ = env.reset()
    waits  = []
    step   = 0
    while step < n_steps:
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            actions = actor.get_dist(obs_t).probs.argmax(dim=-1).numpy()
        obs, _, done, trunc, info = env.step(actions)
        waits.append(info["avg_wait"])
        step += 1
        if done or trunc:
            break
    env.close()
    mean = float(np.mean(waits)) if waits else 0.0
    return waits, mean


def evaluate_fixed_time(cfg, n_steps=500, green_sec=30):
    """
    Evaluate a fixed-time signal controller as a baseline.

    Alternates between NS and EW green phases every green_sec seconds.
    Works on both single-intersection and grid network configs.

    Args:
        cfg:       Path to .sumocfg file.
        n_steps:   Number of simulation steps.
        green_sec: Green phase duration in seconds.

    Returns:
        Tuple of (waits_list, mean_wait).
    """
    traci.start(["sumo", "-c", cfg,
                 "--no-step-log", "true", "--no-warnings", "true"])
    all_tls = traci.trafficlight.getIDList()

    gp_map  = {}
    g2y_map = {}
    timers  = {}
    cur_map = {}

    for tls in all_tls:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
        gp    = [i for i, p in enumerate(logic.phases)
                 if "G" in p.state and "y" not in p.state.lower()]
        if not gp:
            gp = [0]
        gp_map[tls]  = gp
        g2y_map[tls] = {g: g + 1 for g in gp
                        if g + 1 < len(logic.phases)
                        and "y" in logic.phases[g + 1].state.lower()}
        traci.trafficlight.setPhase(tls, gp[0])
        timers[tls]  = 0
        cur_map[tls] = 0

    step  = 0
    waits = []

    while step < n_steps:
        traci.simulationStep()
        step += 1
        w = sum(traci.vehicle.getWaitingTime(v)
                for v in traci.vehicle.getIDList())
        waits.append(w)

        for tls in all_tls:
            timers[tls] += 1
            if timers[tls] >= green_sec and (step + 5) < n_steps:
                cur_g  = gp_map[tls][cur_map[tls]]
                yellow = g2y_map[tls].get(cur_g, cur_g)
                traci.trafficlight.setPhase(tls, yellow)
                for _ in range(4):
                    traci.simulationStep()
                    step += 1
                    waits.append(sum(
                        traci.vehicle.getWaitingTime(v)
                        for v in traci.vehicle.getIDList()))
                cur_map[tls] = (cur_map[tls] + 1) % len(gp_map[tls])
                timers[tls]  = 0
                traci.trafficlight.setPhase(tls, gp_map[tls][cur_map[tls]])

    traci.close()
    mean = float(np.mean(waits)) if waits else 0.0
    return waits, mean
