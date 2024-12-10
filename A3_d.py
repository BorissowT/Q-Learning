import matplotlib
matplotlib.use('Agg')

import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map


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


def plot_learning_curve(env, num_episodes, alpha, epsilon, max_steps, gamma,
                        eval_interval=50):
    Q = np.ones([env.observation_space.n, env.action_space.n])
    success_rates = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        for step in range(max_steps):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, truncated, _ = env.step(action)

            max_next_q = np.max(Q[next_state])

            if reward == -1:
                max_next_q = -10

            if state == next_state:
                reward = -1
                max_next_q = -10

            Q[state, action] += alpha * (
                        reward + gamma * max_next_q - Q[state, action])

            state = next_state

            if done:
                break

        if (episode + 1) % eval_interval == 0:
            print(f"Evaluating at episode {episode + 1}...")
            success_rate = eval_q(Q, env, num_episodes=200, max_steps=max_steps)
            success_rates.append(success_rate)

    plt.plot(range(eval_interval, num_episodes + 1, eval_interval),
             success_rates, marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title(f'Learning Curve (Gamma={gamma})')
    plt.grid(True)

    plt.savefig(f'learning_curve_gamma_{gamma}.png', format='png')
    plt.close()
    print(f"Learning curve plot saved as 'learning_curve_gamma_{gamma}.png'.")


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi",
                   desc=generate_random_map(size=4))

    num_episodes = 3000
    alpha = 0.1
    epsilon = 0.1
    max_steps = 1000
    gamma = 0.99

    plot_learning_curve(env, num_episodes, alpha, epsilon, max_steps, gamma,
                        eval_interval=50)
