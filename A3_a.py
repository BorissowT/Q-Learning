import gym
import numpy as np

from A2_taxi import train_q_learning


def eval_q(Q, env, num_episodes=200, max_steps=1000):

    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        reward = None

        while not done and step < max_steps:
            # take the best option
            action = np.argmax(Q[state])
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            step += 1

        if done and reward == 20:
            successes += 1

    success_rate = successes / num_episodes
    print(
        f"Erfolgsquote: {success_rate * 100:.2f}%"
        f" ({successes}/{num_episodes} erfolgreiche Episoden)")
    return success_rate


if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    # hyperparameters
    num_episodes = 2000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    max_steps = 1000

    Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)

    print("\nEvaluierung der trainierten Policy...")
    eval_q(Q, env)
