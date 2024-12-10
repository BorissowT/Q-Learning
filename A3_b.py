import matplotlib

from A2_frozenlake import train_q_learning

matplotlib.use('TkAgg')
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt


def calculate_success_rate(Q, env, num_tests=100):

    successes = 0
    for _ in range(num_tests):
        state, _ = env.reset()
        done = False
        step = 0
        max_steps = 100

        while not done and step < max_steps:
            action = np.argmax(Q[state])
            state, reward, done, truncated, _ = env.step(action)
            #print(env.render())
            if done and reward > 0:
                successes += 1
            step += 1

    return successes / num_tests


if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="ansi",
        desc=generate_random_map(size=4)
    )

    num_episodes = 5000
    alpha = 0.1
    epsilon = 0.3
    max_steps = 1000
    gamma_values = [0, 0.1, 0.5, 0.9, 1.0]
    success_rates = []

    for gamma in gamma_values:
        print(f"Training with γ = {gamma}...")
        Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)
        success_rate = calculate_success_rate(Q, env)
        success_rates.append(success_rate)
        print(f"Success rate for e = {epsilon} γ = {gamma}: {success_rate:.2f}")

    plt.plot(gamma_values, success_rates, marker='o')
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Successrate')
    plt.title('Hyperparametersearch: Success vs Gamma')
    plt.grid(True)

    plt.savefig(f'gamma_vs_success_rate_e={epsilon}_q_zeros_v3.png', format='png')
    plt.close()
    print("Plot saved as 'gamma_vs_success_rate.png'.")
