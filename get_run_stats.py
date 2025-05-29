baseline = [16.02227, 16.8516, 15.92355]

full = [14.8169, 14.51277, 15.30674]

k2 = [15.35357, 16.4398, 16.35608]

k8 = [15.65488, 16.04286, 16.5367]

k16 = [15.46039, 16.016, 15.67512]

k32 = [15.7806, 16.01338, 14.55087]

k64 = [15.66146, 16.52549, 15.46982]

memory_usages = [5.57, 6.61, 6.61, 6.63, 6.65, 6.73, 12.5]

def get_stats(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return mean, std_dev

def print_stats(label, data):
    mean, std_dev = get_stats(data)
    print(f"{label} - Mean: {mean:.5f}, Std Dev: {std_dev:.5f}")
    
print_stats("Baseline", baseline)
print_stats("Full", full)
print_stats("K2", k2)
print_stats("K8", k8)
print_stats("K16", k16)
print_stats("K32", k32)
print_stats("K64", k64)

# Plot the data where we go from baseline (k=0) to k=64
import matplotlib.pyplot as plt
import numpy as np
k_values = [0, 2, 8, 16, 32, 64, 128]
means = [get_stats(baseline)[0]] + [get_stats(data)[0] for data in [k2, k8, k16, k32, k64]] + [get_stats(full)[0]]
std_devs = [get_stats(baseline)[1]] + [get_stats(data)[1] for data in [k2, k8, k16, k32, k64]] + [get_stats(full)[1]]
print(len(k_values), len(means), len(std_devs))
plt.rcParams.update({'font.size': 14})
# Plot the error bar as a transparent filled area
plt.fill_between(k_values, np.array(means) - np.array(std_devs), np.array(means) + np.array(std_devs), alpha=0.2)
#Print the actual line
# Make the font bigger
# plt.rcParams.update({'font.size': 14})
plt.plot(k_values[:-1], means[:-1], marker='o', color='blue', label='Mean Test MSE')
# Highlight the baseline and full points
plt.scatter([0, 128], [means[0], means[-1]], color='red', label='Baseline and Full', zorder=5)
# Plot a horizontal line at the baseline mean
plt.axhline(y=means[0], color='red', linestyle='--', label='Baseline Mean')
# PLot a dotted line between the 'full' point and the k64 point for both lines
plt.plot([64, 128], [means[-2], means[-1]], linestyle=':', color='blue', alpha=0.5)
plt.xticks(k_values, ['B', 'K2', 'K8', 'K16', 'K32', 'K64', 'Full'])
plt.xlabel('K Value')
plt.ylabel('Test MSE')
plt.title('Test MSE vs K Value')
# Plot memory usage on the same graph
plt.twinx()
plt.plot(k_values[:-1], memory_usages[:-1], marker='x', color='green', label='Memory Usage (GB)')

plt.plot([64, 128], [memory_usages[-2], memory_usages[-1]], linestyle=':', color='green', alpha=0.5)
plt.ylabel('Memory Usage (GB)')
plt.legend(loc='upper left')
plt.legend()
plt.xticks(k_values, ['B', 'K2', 'K8', 'K16', 'K32', 'K64', 'Full'])
plt.grid()
plt.savefig('test_mse_vs_k_value.png')
plt.show()