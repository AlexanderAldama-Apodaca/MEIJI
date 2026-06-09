import random
import numpy as np

from colonial_diplomacy_environment import Order
from colonial_diplomacy_environment import ColonialDiplomacyEnv

class RandomColonialAgent:
    def generate_orders(self, env, player_id):
        orders = []

        units = env.units.get(player_id, [])

        for unit in units:
            location = unit["location"]

            candidates = env.adjacency.get(location, [])

            # Hold probability
            if random.random() < 0.3:
                orders.append(
                    Order(
                        player_id=player_id,
                        unit_location=location,
                        order_type="HOLD"
                    )
                )

                continue

            legal_moves = []

            for target in candidates:
                if env.can_unit_move_to(unit["type"], target):
                    legal_moves.append(target)

            if not legal_moves:
                orders.append(
                    Order(
                        player_id=player_id,
                        unit_location=location,
                        order_type="HOLD"
                    )
                )
                
                continue

            destination = random.choice(legal_moves)

            orders.append(
                Order(
                    player_id=player_id,
                    unit_location=location,
                    order_type="MOVE",
                    target=destination
                )
            )

        return orders
    
def run_game(max_turns=100, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        env = ColonialDiplomacyEnv()

        agents = {
            pid: RandomColonialAgent()
            for pid in range(1, env.num_players + 1)
        }

        turn = 0
        done = False

        while not done and turn < max_turns:
            joint_orders = {}

            for pid, agent in agents.items():
                joint_orders[pid] = agent.generate_orders(env, pid)

            observations, rewards, done, info = env.step(joint_orders)

            turn += 1

        winner = env.check_victory()

        return {
            "env": env,
            "turns": turn,
            "winner": env.check_victory(),
            "history": env.history
        }
