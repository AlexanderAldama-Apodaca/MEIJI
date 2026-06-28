import random
import numpy as np

from colonial_diplomacy_environment import Order
from colonial_diplomacy_environment import ColonialDiplomacyEnv

class RandomColonialAgent:
    def generate_candidate_orders(self, env, player_id, unit):
        orders = []

        location = unit["location"]

        candidates = env.adjacency.get(location, [])

        # HOLD
        orders.append(Order(player_id=player_id, unit_location=location, order_type="HOLD"))

        # MOVE
        for target in candidates:
            if env.can_unit_move_to(unit["type"], target):
                orders.append(Order(player_id=player_id, unit_location=location, order_type="MOVE", target=target))

        # SUPPORT HOLD
        for friendly_unit in env.units.get(player_id, []):
            support_location = friendly_unit["location"]

            if support_location == location:
                continue

            if support_location in candidates:
                orders.append(Order(player_id=player_id, unit_location=location, order_type="SUPPORT", support_unit=support_location, support_target=support_location))

        # SUPPORT MOVE
        for friendly_unit in env.units.get(player_id, []):
            support_origin = friendly_unit["location"]

            if support_origin == location:
                continue

            for support_destination in env.adjacency.get(support_origin, []):
                orders.append(Order(player_id=player_id, unit_location=location, order_type="SUPPORT", support_unit=support_origin, support_target=support_destination))

        # CONVOY
        if unit["type"] == "Fleet":
            for friendly_unit in env.units.get(player_id, []):
                if friendly_unit["type"] != "Army":
                    continue

                army_location = friendly_unit["location"]

                for destination in env.adjacency.get(army_location, []):
                    orders.append(Order(player_id=player_id, unit_location=location, order_type="CONVOY", convoy_origin=army_location, convoy_destination=destination))

        return orders

    def generate_adjustment_orders(self, env, player_id):
        orders = []

        unit_count = len(env.units.get(player_id, []))

        supply_count = len(env.controlled_supply_centers[player_id])

        adjustment = supply_count - unit_count

        if adjustment > 0:
            for province, owner in (env.home_supply_centers.items()):
                if owner != player_id:
                    continue

                if province not in (env.controlled_supply_centers[player_id]):
                    continue

                pid, occupying_unit = (env.get_unit_at(province))

                if occupying_unit is not None:
                    continue

                orders.append(Order(player_id=player_id, unit_location=province, order_type="BUILD", target=province, build_type="Army"))

                if env.is_coastal(province):
                    orders.append(Order(player_id=player_id, unit_location=province, order_type="BUILD", target=province, build_type="Fleet"))

        elif adjustment < 0:
            for unit in env.units[player_id]:
                orders.append(Order(player_id=player_id, unit_location=unit["location"], order_type="DISBAND", target=unit["location"]))

        return orders
    
    def generate_retreat_orders(self, env, player_id):
        orders = []

        for retreat in env.pending_retreats:
            if retreat["player_id"] != player_id:
                continue

            unit = retreat["unit"]

            location = unit["location"]

            for destination in retreat["retreat_options"]:
                orders.append(Order(player_id=player_id, unit_location=location, order_type="RETREAT", target=destination))

            orders.append(Order(player_id=player_id, unit_location=location, order_type="DISBAND", target=location))

        return orders

    def generate_orders(self, env, player_id):
        if env.phase == "Retreat":
            orders = []

            for retreat in env.pending_retreats:
                if retreat["player_id"] != player_id:
                    continue

                retreat_candidates = []

                location = retreat["unit"]["location"]

                for destination in retreat["retreat_options"]:
                    retreat_candidates.append(Order(player_id=player_id, unit_location=location, order_type="RETREAT", target=destination))

                retreat_candidates.append(Order(player_id=player_id, unit_location=location, order_type="DISBAND", target=location))

                orders.append(random.choice(retreat_candidates))

            return orders

        if env.phase == "Winter_Adjustment":
            candidates = self.generate_adjustment_orders(env, player_id)

            if not candidates:
                return []
            
            unit_count = len(env.units[player_id])
            supply_count = len(env.controlled_supply_centers[player_id])

            adjustment = supply_count - unit_count

            orders = []

            if adjustment > 0:
                builds = [o for o in candidates if o.order_type == "BUILD"]

                random.shuffle(builds)

                orders.extend(builds[:adjustment])

            elif adjustment < 0:
                disbands = [o for o in candidates if o.order_type == "DISBAND"]

                random.shuffle(disbands)

                orders.extend(disbands[:abs(adjustment)])

            return orders
        
        orders = []

        units = env.units.get(player_id, [])

        for unit in units:
            candidates = (self.generate_candidate_orders(env, player_id, unit))

            chosen_order = random.choice(candidates)

            orders.append(chosen_order)

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

    def evaluate_order(self, env, player_id, order):
        score = 0

        if order.order_type == "HOLD":
            score += 1

        elif order.order_type == "MOVE":
            if order.target in env.supply_centers:
                score += 10
            else:
                score += 2

        elif order.order_type == "SUPPORT":
            if order.support_unit == order.support_target:
                score += 5
            else:
                score += 7

        elif order.order_type == "CONVOY":
            score += 4

        elif order.order_type == "RETREAT":
            if order.target in env.supply_centers:
                score += 9
            else:
                score += 6

        elif order.order_type == "BUILD":
            if order.build_type == "Army":
                score += 8

            elif order.build_type == "Fleet":
                score += 8

        elif order.order_type == "DISBAND":
            score += 3

        return score

    def generate_candidate_orders(self, env, player_id, unit):
        orders = []

        location = unit["location"]

        candidates = env.adjacency.get(location, [])

        # HOLD
        orders.append(Order(player_id=player_id, unit_location=location, order_type="HOLD"))

        # MOVE
        for target in candidates:
            if env.can_unit_move_to(unit["type"], target):
                orders.append(Order(player_id=player_id, unit_location=location, order_type="MOVE", target=target))

        # SUPPORT HOLD
        for friendly_unit in env.units.get(player_id, []):
            support_location = friendly_unit["location"]

            if support_location == location:
                continue

            if support_location in candidates:
                orders.append(Order(player_id=player_id, unit_location=location, order_type="SUPPORT", support_unit=support_location, support_target=support_location))

        # SUPPORT MOVE
        for friendly_unit in env.units.get(player_id, []):
            support_origin = friendly_unit["location"]

            if support_origin == location:
                continue

            for support_destination in env.adjacency.get(support_origin, []):
                orders.append(Order(player_id=player_id, unit_location=location, order_type="SUPPORT", support_unit=support_origin, support_target=support_destination))

        # CONVOY
        if unit["type"] == "Fleet":
            for friendly_unit in env.units.get(player_id, []):
                if friendly_unit["type"] != "Army":
                    continue

                army_location = friendly_unit["location"]

                for destination in env.adjacency.get(army_location, []):
                    orders.append(Order(player_id=player_id, unit_location=location, order_type="CONVOY", convoy_origin=army_location, convoy_destination=destination))

        return orders
    
    def generate_adjustment_orders(self, env, player_id):
        orders = []

        unit_count = len(env.units.get(player_id, []))

        supply_count = len(env.controlled_supply_centers[player_id])

        adjustment = supply_count - unit_count

        if adjustment > 0:
            for province, owner in env.home_supply_centers.items():
                if owner != player_id:
                    continue

                if province not in env.controlled_supply_centers[player_id]:
                    continue

                pid, occupying_unit = env.get_unit_at(province)

                if occupying_unit is not None:
                    continue

                orders.append(Order(player_id=player_id, unit_location=province, order_type="BUILD", target=province, build_type="Army"))

                if env.is_coastal(province):
                    orders.append(Order(player_id=player_id, unit_location=province, order_type="BUILD", target=province, build_type="Fleet"))

        elif adjustment < 0:
            for unit in env.units[player_id]:
                orders.append(Order(player_id=player_id, unit_location=unit["location"], order_type="DISBAND", target=unit["location"]))

        return orders
    
    def generate_retreat_orders(self, env, player_id):
        orders = []

        for retreat in env.pending_retreats:
            if retreat["player_id"] != player_id:
                continue

            unit = retreat["unit"]

            location = unit["location"]

            for destination in retreat["retreat_options"]:
                orders.append(Order(player_id=player_id, unit_location=location, order_type="RETREAT", target=destination))

            orders.append(Order(player_id=player_id, unit_location=location, order_type="DISBAND", target=location))

        return orders

    def generate_orders(self, env, player_id):
        # Exploration
        if random.random() < self.epsilon:
            return RandomColonialAgent().generate_orders(env, player_id)
        
        orders = []

        if env.phase == "Retreat":
            orders = []

            for retreat in env.pending_retreats:
                if retreat["player_id"] != player_id:
                    continue

                retreat_candidates = []

                location = retreat["unit"]["location"]

                for destination in retreat["retreat_options"]:
                    retreat_candidates.append(Order(player_id=player_id, unit_location=location, order_type="RETREAT", target=destination))

                retreat_candidates.append(Order(player_id=player_id, unit_location=location, order_type="DISBAND", target=location))

                best_order = None
                best_score = float("-inf")

                for order in retreat_candidates:
                    score = self.evaluate_order(env, player_id, order)

                    if score > best_score:
                        best_score = score
                        best_order = order

                orders.append(best_order)

            return orders

        if env.phase == "Winter_Adjustment":
            candidate_orders = self.generate_adjustment_orders(env, player_id)

            if not candidate_orders:
                return []
            
            unit_count = len(env.units[player_id])
            supply_count = len(env.controlled_supply_centers[player_id])

            adjustment = supply_count - unit_count

            orders = []

            remaining_candidates = candidate_orders.copy()

            if adjustment > 0:
                for _ in range(adjustment):
                    build_candidates = [o for o in remaining_candidates if o.order_type == "BUILD"]

                    if not build_candidates:
                        break

                    best_order = None
                    best_score = float("-inf")

                    for order in build_candidates:
                        score = self.evaluate_order(env, player_id, order)

                        if score > best_score:
                            best_score = score
                            best_order = order

                    orders.append(best_order)

                    remaining_candidates = [candidate for candidate in remaining_candidates if not (candidate.order_type == "BUILD" and candidate.target == best_order.target)]

            elif adjustment < 0:
                for _ in range(abs(adjustment)):
                    disband_candidates = [o for o in remaining_candidates if o.order_type == "DISBAND"]

                    if not disband_candidates:
                        break

                    best_order = None
                    best_score = float("-inf")

                    for order in disband_candidates:
                        score = self.evaluate_order(env, player_id, order)

                        if score > best_score:
                            best_score = score
                            best_order = order

                    orders.append(best_order)

                    remaining_candidates.remove(best_order)

            return orders

        units = env.units.get(player_id, [])

        for unit in units:
            candidate_orders = self.generate_candidate_orders(env, player_id, unit)

            best_order = None
            best_score = float("-inf")

            for order in candidate_orders:
                score = self.evaluate_order(env, player_id, order)

                if score > best_score:
                    best_score = score
                    best_order = order

            orders.append(best_order)

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
                previous_states[pid] = agent.get_state_key(env, pid)

            joint_orders = {}

            for pid, agent in agents.items():
                joint_orders[pid] = agent.generate_orders(env, pid)

            observations, rewards, done, info = env.step(joint_orders)

            for pid, agent in agents.items():
                next_state = agent.get_state_key(env, pid)

                reward = rewards[pid]

                agent.update(previous_states[pid], reward, next_state)

            turn += 1

        winner = env.check_victory()

        return {
            "turns": turn,
            "winner": winner,
            "history": env.history
        }
