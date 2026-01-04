import numpy as np
import random

state = env.reset()
next_state, reward, done, info = env.step(action)

def epsilon_greedy(Q, state, n_actions, epsilon):
  if random.random() < epsilon:
    return random.randrange(n_actions)
  return np.argmax(Q[state])

def create_q_table(n_states, n_actions):
  return np.zeros((n_states, n_actions))

def create_v_table(n_states):
  return np.zeros(n_states)

# Tabular TD(0) - Value Prediction
def td0_prediction(env, policy, gamma=0.99, alpha=0.1, episodes=500):
    V = create_v_table(env.observation_space.n)

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            state = next_state

    return V

# Sarsa - On-Policy TD Control
def sarsa(env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=500):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = create_q_table(n_states, n_actions)

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, n_actions, epsilon)
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)

            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            action = next_action

    return Q

# Q-learning - Off-Policy TD Control
def q_learning(env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=500):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = create_q_table(n_states, n_actions)

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy(Q, state, n_actions, epsilon)
            next_state, reward, done, _ = env.step(action)

            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

    return Q

# Double Q-learning
def double_q_learning(env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=500):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    Q1 = create_q_table(n_states, n_actions)
    Q2 = create_q_table(n_states, n_actions)

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            Q_sum = Q1 + Q2
            action = epsilon_greedy(Q_sum, state, n_actions, epsilon)
            
            next_state, reward, done, _ = env.step(action)

            if random.random() < 0.5:
                best_next = np.argmax(Q1[next_state])
                td_target = reward + gamma * Q2[next_state][best_next] * (not done)
                td_error = td_target - Q1[state][action]
                Q1[state][action] += alpha * td_error
            else:
                best_next = np.argmax(Q2[next_state])
                td_target = reward + gamma * Q1[next_state][best_next] * (not done)
                td_error = td_target - Q2[state][action]
                Q2[state][action] += alpha * td_error

            state = next_state

    return Q1, Q2
