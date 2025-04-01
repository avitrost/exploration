import matplotlib.pyplot as plt
import numpy as np

# Given data
ppo_search = np.load('reward_counts_by_epoch_1.npy')
ppo_search[0] = 0

bon = np.load('first_correct_counts_1024_aime.npy')
bon = bon[:129]
bon[0] = 0

# Compute cumulative sums to ensure non-decreasing values
y_ppo = np.cumsum(ppo_search)
y_bon = np.cumsum(bon)

# Create corresponding x-axis values
x_ppo = np.arange(len(ppo_search))
x_bon = np.arange(len(bon))

# Plotting both datasets on the same figure
plt.figure(figsize=(8, 5))
plt.plot(x_ppo, y_ppo, marker='o', linestyle='-', color='b', markersize=4, label='ppo_search')
plt.plot(x_bon, y_bon, marker='o', linestyle='-', color='r', markersize=4, label='bon')

# Set labels, title, grid and legend
plt.xlabel("# Generations Per Example (k)")
plt.ylabel("Pass@k")
plt.title("Pass@k with Qwen2.5-3B On AIME24")
plt.ylim(0, 30)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

print(ppo_search.sum() / 30, bon.sum() / 30)
