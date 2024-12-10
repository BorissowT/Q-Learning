import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map

from A2_frozenlake import train_q_learning


def eval_q(Q, env, num_episodes=200, max_steps=1000):

    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        reward = None

        while not done and step < max_steps:
            action = np.argmax(Q[state])
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            step += 1

        if done and reward == 1:
            successes += 1

    success_rate = successes / num_episodes
    print(
        f"Success Rate: {success_rate * 100:.2f}%"
        f" ({successes}/{num_episodes} successful episodes)")
    return success_rate


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True,
                   desc=generate_random_map(size=4))

    num_episodes = 15000
    alpha = 0.25
    gamma = 0.95
    epsilon = 0.5
    max_steps = 1000

    Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)

    print("\nEvaluating policy...")
    eval_q(Q, env)
