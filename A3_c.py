import matplotlib
from A2_frozenlake import train_q_learning

matplotlib.use('TkAgg')
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


def hyperparameter_search(env, num_episodes, alpha, epsilon, max_steps,
                          gamma_values):
    success_rates = []

    for gamma in gamma_values:
        print(f"Testing gamma = {gamma}...")
        Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon,
                             max_steps)
        total_reward = test_q(Q, env)
        success_rate = total_reward / max_steps
        success_rates.append(success_rate)

    return success_rates


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

    num_episodes = 15000
    alpha = 0.25
    epsilon = 0.5
    max_steps = 1000
    gamma_values = [0, 0.1, 0.5, 0.9, 1]

    all_success_rates = []

    for i in range(10):
        print(f"\nRunning hyperparameter search experiment {i+1}...")
        success_rates = hyperparameter_search(env, num_episodes, alpha, epsilon,
                                              max_steps, gamma_values)
        all_success_rates.append(success_rates)
        print(f"Success rates for experiment {i+1}: {success_rates}")

    all_success_rates = np.array(all_success_rates)

    mean_success_rates = all_success_rates.mean(axis=0)

    plt.plot(gamma_values, mean_success_rates, marker='o')
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Average Success Rate')
    plt.title('Hyperparameter Search: Success vs Gamma '
              '(Average over 10 experiments)')
    plt.grid(True)

    plt.savefig(f'gamma_vs_average_success_rate_slippery_e={epsilon}_q_zeros_v1.png', format='png')
    plt.close()
