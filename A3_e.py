import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gym

from A2_taxi import train_q_learning
from A3_a_taxi import eval_q


def hyperparameter_search_taxi(gammas,
                               alpha,
                               epsilon,
                               num_episodes,
                               max_steps,
                               eval_episodes=200):

    success_rates = {}
    env = gym.make("Taxi-v3")

    for gamma in gammas:
        print(f"Training for γ = {gamma}...")
        Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)
        success_rate = eval_q(Q, env, num_episodes=eval_episodes)
        success_rates[gamma] = success_rate

    env.close()
    return success_rates


if __name__ == "__main__":
    alpha = 0.1
    epsilon = 0.1
    num_episodes = 5000
    max_steps = 1000
    gammas = [0, 0.1, 0.5, 0.9, 1]

    print("Optimize for Taxi...")
    taxi_results = hyperparameter_search_taxi(gammas, alpha, epsilon,
                                              num_episodes, max_steps)

    plt.figure(figsize=(8, 6))
    plt.plot(list(taxi_results.keys()), list(taxi_results.values()), marker='o',
             label="Taxi", color='orange')
    plt.title("Taxi: Success for γ")
    plt.xlabel("γ")
    plt.ylabel("Successrate")
    plt.grid()
    plt.legend()
    plt.savefig("taxi_hyperparameter_search.png")
