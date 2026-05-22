import json

def load_game(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    
def summarize_phase(phase):
    return{
        "phase": phase["name"],
        "num_orders": sum(len(v) for v in phase["orders"].values()),
        "num_results": len(phase["results"]),
        "num_players": len(phase["state"]["units"])
    }

def summarize_game(filepath):
    game = load_game(filepath)

    summaries = []

    for phase in game["phases"]:
        summaries.append(summarize_phase(phase))

    return summaries
