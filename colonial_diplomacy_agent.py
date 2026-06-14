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
            "turns": turn,
            "winner": env.check_victory(),
            "history": env.history
        }

class TDColonialAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # State-value table
        self.value_table = {}
    
    def get_state_key(self, env, player_id):
        owned_centers = env.count_supply_centers(player_id)
        
        unit_locations = sorted(unit["location"] for unit in env.units.get(player_id, []))

        return (owned_centers, tuple(unit_locations))
    
    def get_value(self, state_key):
        return self.value_table.get(state_key, 0.0)
    
    def update(self, state, reward, next_state):
        current_value = self.get_value(state)

        next_value = self.get_value(next_state)

        td_target = reward + self.gamma * next_value

        td_error = td_target - current_value

        self.value_table[state] = (current_value + self.alpha * td_error)

    def generate_orders(self, env, player_id):
        # Exploration
        if random.random() < self.epsilon:
            return RandomColonialAgent().generate_orders(env, player_id)
        
        orders = []

        units = env.units.get(player_id, [])

        for unit in units:
            location = unit["location"]

            candidates = env.adjacency.get(location, [])

            legal_moves = []

            for target in candidates:
                if env.can_unit_move_to(unit["type"], target):
                    legal_moves.append(target)

            if not legal_moves:
                orders.append(Order(player_id=player_id, unit_location=location, order_type="HOLD"))
                continue

            best_target = None
            best_score = float("-inf")

            for target in legal_moves:
                score = 0

                if target in env.supply_centers:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_target = target

            orders.append(Order(player_id=player_id, unit_location=location, order_type="MOVE", target=best_target))

        return orders
    
    def run_td_game(max_turns=100, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        env = ColonialDiplomacyEnv()

        agents = {
            pid: TDColonialAgent()
            for pid in range(1, env.num_players + 1)
        }

        turn = 0
        done = False

        while not done and turn < max_turns:
            previous_states = {}

            for pid, agent in agents.items():
                previous_states[pid] = (agent.get_state_key(env, pid))

            joint_orders = {}

            for pid, agent in agents.items():
                joint_orders[pid] = (agent.generate_orders(env, pid))

            observations, rewards, done, info = (env.step(joint_orders))

            for pid, agent in agents.items():
                next_state = (agent.get_state_key(env, pid))

                reward = rewards[pid]

                agent.update(previous_states[pid], reward, next_state)

            turn += 1

        return {
            "turns": turn,
            "winner": env.check_victory(),
            "history": env.history
        }
