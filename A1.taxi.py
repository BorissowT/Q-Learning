import gym

env = gym.make("Taxi-v3", render_mode="ansi")

state, info = env.reset()
print(f"Initial state: {state}")
print(f"Info: {info}")

print(env.render())

# 0: move south
# 1: move north
# 2: move east
# 3: move west
# 4: pickup passenger
# 5: drop off passenger

next_state, reward, done, truncated, info = env.step(1)
print(f"Next state: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")

print(env.render())
