import json
import os
import traceback
from datetime import datetime

from colonial_diplomacy_agent import run_game

RAW_GAME_DIR = "data/raw_games"
LOG_DIR = "data/logs"

os.makedirs(RAW_GAME_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def validate_game(result):
    """
    Basic sanity checks.
    """

    history = result["history"]

    if len(history) == 0:
        raise ValueError("Game produced empty history.")
    
    return True

def save_game(game_id, result):
    """
    Save a complete game record.
    """

    filepath = os.path.join(RAW_GAME_DIR, f"game_{game_id:06d}.json")

    export = {
        "game_id": game_id,
        "winner": result["winner"],
        "turns": result["turns"],
        "history": result["history"]
    }

    with open(filepath, "w") as f:
        json.dump(export, f, indent=2)

def log_crash(game_id, seed, exc):
    crash_file = os.path.join(LOG_DIR, "crashes.log")

    with open(crash_file, "a") as f:
        f.write("\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(
            f"{datetime.utcnow()} | "
            f"Game={game_id} | Seed={seed}\n"
        )

def initialize_stats():
    return{
        "games_completed": 0,
        "games_crashed": 0,
        "victory_games": 0,
        "draw_games": 0,
        "winner_counts": {},
        "turn_counts": []
    }

def update_stats(stats, result):
    stats["games_completed"] += 1

    stats["turn_counts"].append(result["turns"])

    winner = result["winner"]

    if winner is None:
        stats["draw_games"] += 1
    else:
        stats["victory_games"] += 1

        winner = str(winner)

        if winner not in stats["winner_counts"]:
            stats["winner_counts"][winner] = 0

        stats["winner_counts"][winner] += 1

def finalize_stats(stats):
    if stats["turn_counts"]:
        stats["average_turns"] = (sum(stats["turn_counts"]) / len(stats["turn_counts"]))

        stats["max_turns"] = max(stats["turn_counts"])
        stats["min_turns"] = min(stats["turn_counts"])
    else:
        stats["average_turns"] = 0
        stats["max_turns"] = 0
        stats["min_turns"] = 0

    return stats

def save_stats(stats):
    stats_file = os.path.join(LOG_DIR, "dataset_statistics.json")

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

def generate_dataset(num_games=1000, max_turns=200):
    stats = initialize_stats()

    for game_id in range(num_games):
        seed = game_id
        
        try:
            result = run_game(max_turns=max_turns, seed=seed)

            validate_game(result)

            save_game(game_id, result)

            update_stats(stats, result)

        except Exception as exc:
            stats["games_crashed"] += 1

            log_crash(game_id, seed, exc)

        if (game_id + 1) % 100 == 0:
            print(
                f"[{game_id + 1}/{num_games}] "
                f"Completed={stats['games_completed']} "
                f"Crashes={stats['games_crashed']}"
            )

    stats = finalize_stats(stats)

    save_stats(stats)

    print("\nDataset generation complete.")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    generate_dataset(num_games=1000, max_turns=200)
