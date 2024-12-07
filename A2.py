import numpy as np
import gym


def update_q(Q, state, action, reward, next_state, alpha, gamma):
    max_next_q = np.max(
        Q[next_state])
    Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * max_next_q - Q[state, action])


def epsilon_greedy_action(Q, state, epsilon, num_actions):
    # try random action
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)
    return np.argmax(Q[state])


def train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # fill array with zeros
    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action = epsilon_greedy_action(Q, state, epsilon, num_actions)
            next_state, reward, done, truncated, _ = env.step(action)

            update_q(Q, state, action, reward, next_state, alpha, gamma)

            state = next_state
            step += 1

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} abgeschlossen.")

    return Q


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="ansi")

    # hyperparameters
    num_episodes = 5000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    max_steps = 1000

    Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)

    state, _ = env.reset()
    done = False
    total_reward = 0

    print("\nTraining abgeschlossen. Teste die Policy...")
    while not done:
        action = np.argmax(
            Q[state])
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        print(env.render())

    print(f"Gesamtbelohnung: {total_reward}")
