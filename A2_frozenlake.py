import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def update_q(Q, state, action, reward, next_state, alpha, gamma):
    """
    Aktualisiert das Q-Array basierend auf der Q-Learning-Regel.
    Wenn der Agent in ein Loch tritt, wird der Q-Wert stärker bestraft.
    """
    max_next_q = np.max(Q[next_state])  # Maximaler Q-Wert im Folgezustand

    if reward == -1:
        max_next_q = -10

    if state == next_state:
        reward = -1

    Q[state, action] += alpha * (reward + gamma * max_next_q - Q[state, action])


def epsilon_greedy_action(Q, state, epsilon, num_actions):
    """
    Wählt eine Aktion epsilon-greedy aus dem aktuellen Q-Array.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(
            num_actions)  # Zufällige Aktion mit Wahrscheinlichkeit epsilon
    return np.argmax(Q[state])  # Beste Aktion basierend auf Q-Werten


def train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps):
    """
    Führt das Q-Learning für eine gegebene Umgebung aus.
    """
    num_states = env.observation_space.n  # Anzahl der Zustände
    num_actions = env.action_space.n  # Anzahl der Aktionen

    # Initialisiere das Q-Array mit Einsen
    Q = np.ones(
        (num_states, num_actions))  # Statt mit Nullen beginnen wir mit Einsen

    for episode in range(num_episodes):
        state_before, _ = env.reset()  # Setze die Umgebung zurück
        done = False
        step = 0

        while not done and step < max_steps:
            action = epsilon_greedy_action(Q, state_before, epsilon, num_actions)
            next_state, reward, done, truncated, _ = env.step(action)

            # Q-Update
            update_q(Q, state_before, action, reward, next_state, alpha, gamma)

            state_before = next_state
            step += 1

        # Optional: Fortschrittsanzeige alle 100 Episoden
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} abgeschlossen.")

    return Q


def test_q(Q):
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
    total_reward = 0
    step = 0

    max_steps = 100
    while not done and step < max_steps:
        action = np.argmax(
            Q[state])  # Wähle die beste Aktion basierend auf den Q-Werten
        state, reward, done, truncated, _ = env.step(
            action)  # Führe die Aktion aus und beobachte das Ergebnis
        total_reward += reward
        step += 1
        print(env.render())  # Gebe die Umgebung nach jeder Aktion aus

    print(f"\nTotal reward during testing: {total_reward}")


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
    epsilon = 0.1
    max_steps = 1000

    # Train the agent using Q-learning
    Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, max_steps)

    print("\nTraining is done. Now testing the agent:")
    test_q(Q)
