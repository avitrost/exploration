import matplotlib.pyplot as plt
import numpy as np

# Given data
ppo_search = np.load('ppo_search_qwen2.5_3b_math_500.npy')
ppo_search[0] = 0

bon = np.load('1024_bon_qwen2.5_3b_math_500.npy')
# bon = bon[:129]
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
plt.xlabel("# Generations Per Example")
plt.ylabel("Cumulative Count of Examples Receiving Reward")
plt.title("Cumulative Distribution of Reward Reception by # Generations with Qwen2.5 3B On MATH-500")
plt.ylim(0, 500)
plt.grid(True)
plt.legend()

# Show the plot
# plt.show()

print(ppo_search.sum() / 500, bon.sum() / 500)
