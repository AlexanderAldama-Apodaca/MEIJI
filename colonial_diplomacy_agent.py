import random

class TDAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = {} # {(obs, action): value}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, obs, action):
        return self.Q.get((obs, action), 0.0)

    def select_action(self, env, pid, obs):
        if random.random() < self.epsilon:
            return random_action(env, pid)
        return best_action(self, env, pid, obs)
    
    def update(self, obs, action, reward, next_obs, next_action):
        old = self.get_q(obs, action)
        target = reward + self.gamma * self.get_q(next_obs, next_action)
        self.Q[(obs, action)] = old + self.alpha * (target - old)

def get_observation(env, pid):
    my_sc = env.count_supply_centers(pid)

    obs = [my_sc]

    for other in range(1, env.num_players + 1):
        if other == pid:
            continue

        other_sc = env.count_supply_centers(other)

        if other_sc < my_sc:
            obs.append(-1)
        elif other_sc == my_sc:
            obs.append(0)
        else:
            obs.append(1)

    return tuple(obs)
    
def train_multi_agent(env, episodes=500):
    agents = {
        pid: TDAgent()
        for pid in range(1, env.num_players + 1)
    }

    for episode in range(episodes):
        env.__init__() # reset environment properly

        # initial observations
        obs = {
            pid: get_observation(env, pid)
            for pid in agents
        }

        # initial actions
        actions = {
            pid: agents[pid].select_action(env, pid, obs[pid])
            for pid in agents
        }

        done = False

        while not done:
            # execute full turn
            next_obs, rewards, done, _ = env.step(actions)

            # choose next actions
            next_actions = {
                pid: agents[pid].select_action(env, pid, next_obs[pid])
                for pid in agents
            }

            # to update
            for pid in agents:
                agents[pid].update(obs[pid], actions[pid], rewards[pid], next_obs[pid], next_actions[pid])
            
            # advance
            obs = next_obs
            actions = next_actions

    return agents
    
def random_action(env, pid):
    units = env.units.get(pid, [])

    if not units:
        return (0, 0, None)
    
    unit_idx = random.randrange(len(units))
    move_type = random.choice([0, 1]) # hold or move

    if move_type == 0:
        return (unit_idx, 0, None)
    
    loc = units[unit_idx]["location"]
    neighbors = env.adjacency.get(loc, [])

    if not neighbors:
        return (unit_idx, 0, None)
    
    target = random.choice(neighbors)
    return (unit_idx, 1, target)
    
def best_action(agent, env, pid, obs):
    units = env.units.get(pid, [])

    if not units:
        return (0, 0, None)
    
    candidates = []

    for i, unit in enumerate(units):
        loc = unit["location"]

        # hold
        candidates.append((i, 0, None))

        # moves
        for n in env.adjacency.get(loc, []):
            candidates.append((i, 1, n))

    best = None
    best_val = -float("inf")

    for a in candidates:
        val = agent.get_q(obs, a)
        if val > best_val:
            best_val = val
            best = a

    return best if best else random.choice(candidates)
