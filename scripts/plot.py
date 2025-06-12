import matplotlib.pyplot as plt
import numpy as np

# Given data
ppo_search = np.load('pass@k_ppo-aime-Qwen2.5-Math-7B-0.npy')
ppo_search[0] = 0

bon = np.load('first_correct_counts_128_aime_28.npy')
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
plt.xlabel("k")
plt.ylabel("Pass@k")
plt.title("Pass@k with Qwen2.5-Math-7B On AIME24 (28/30 subset)")
plt.ylim(0, 28)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

print(ppo_search.sum() / 28, bon.sum() / 28)
