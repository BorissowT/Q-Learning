import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def update_q(Q, state, action, reward, next_state, alpha, gamma):
    max_next_q = np.max(Q[next_state])

    if reward == -1:
        max_next_q = -10

    if state == next_state:
        reward = -1

    Q[state, action] += alpha * (reward + gamma * max_next_q - Q[state, action])

def epsilon_greedy_action(Q, state, epsilon, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)
    return np.argmax(Q[state])


def train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    Q = np.ones((num_states, num_actions))

    for episode in range(num_episodes):
        state_before, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action = epsilon_greedy_action(Q, state_before, epsilon,
                                           num_actions)
            next_state, reward, done, truncated, _ = env.step(action)

            update_q(Q, state_before, action, reward, next_state, alpha, gamma)

            state_before = next_state
            step += 1

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} abgeschlossen.")

    return Q


def test_q(Q, env):
    """
    Test the trained agent

    0: Move left
    1: Move down
    2: Move right
    3: Move up

    MAP of states:
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15
    """
    state, _ = env.reset()
    print(env.render())

    done = False
    step = 0

    max_steps = 100
    while not done and step < max_steps:
        action = np.argmax(
            Q[state])
        state, reward, done, truncated, _ = env.step(
            action)
        step += 1
        print(env.render())


def test_into_wall(Q):
    """
    Test agent bumps into wall

    0: Move left
    1: Move down
    2: Move right
    3: Move up

    MAP of states:
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15
    """
    state_before, _ = env.reset()

    print(env.render())
    action = np.argmax(0)
    state_after, reward, done, truncated, _ = env.step(action)
    print(env.render())


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1",
                   is_slippery=False,
                   render_mode="ansi",
                   desc=generate_random_map(size=4))

    num_episodes = 5000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.2
    max_steps = 1000

    Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)

    print("\nTraining is done. Now testing the agent:")
    test_q(Q, env)
