import json

def load_game(filepath):
    with open(filepath, "r") as diplomacyFile:
        return json.load(diplomacyFile)
    
def get_power_order(supply_centers):
    """
    Returns a consistent global ordering of powers.
    """
    return sorted(supply_centers.keys())

def compute_sc_counts(supply_centers):
    """
    Converts centers dict -> {power: count}
    """
    return {p: len(centers) for p, centers in supply_centers.items()}

def relative_relation(a, b):
    """
    Compare two supply center counts.
    """
    if b < a:
        return -1
    elif b == a:
        return 0
    else:
        return 1
    
def build_observation_for_agent(agent, sc_counts, power_order):
    """
    Build observation vector for a single agent.
    """
    agent_sc = sc_counts[agent]

    obs = [agent_sc]

    for other in power_order:
        if other == agent:
            continue
        rel = relative_relation(agent_sc, sc_counts[other])
        obs.append(rel)

    return obs

def convert_phase_to_observations(phase):
    """
    Convert one phase into observations for all agents.
    """
    centers = phase["state"]["centers"]

    power_order = get_power_order(centers)
    sc_counts = compute_sc_counts(centers)

    obs = {}
    for agent in power_order:
        obs[agent] = build_observation_for_agent(agent, sc_counts, power_order)

    return obs

def convert_game(filepath):
    game = load_game(filepath)

    all_phase_obs = []

    for phase in game["phases"]:
        phase_obs = {
            "phase": phase["name"],
            "observations": convert_phase_to_observations
        }
        all_phase_obs.append(phase_obs)

    return all_phase_obs
