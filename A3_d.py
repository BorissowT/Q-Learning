import matplotlib
matplotlib.use('Agg')

import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map


def test_q(Q, env):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    max_steps = 100
    while not done and step < max_steps:
        action = np.argmax(Q[state])
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
    return total_reward


def plot_learning_curve(env, num_episodes, alpha, epsilon, max_steps, gamma,
                        eval_interval=50):
    Q = np.ones([env.observation_space.n, env.action_space.n])
    success_rates = []

    # Training loop
    for episode in range(num_episodes):
        # Run one episode of training
        state, _ = env.reset()
        done = False
        for step in range(max_steps):
            # Choose action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                action = np.argmax(Q[state])  # Exploitation

            # Take action and update Q
            next_state, reward, done, truncated, _ = env.step(action)
            Q[state, action] += alpha * (
                        reward + gamma * np.max(Q[next_state]) - Q[
                    state, action])
            state = next_state

            if done:
                break

        # Evaluate every `eval_interval` episodes
        if (episode + 1) % eval_interval == 0:
            total_reward = test_q(Q, env)
            success_rate = total_reward / max_steps
            success_rates.append(success_rate)

    # Plot the learning curve
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
