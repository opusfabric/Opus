import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from seaborn import color_palette

colors = color_palette("Set2", 3)


from matplotlib.ticker import ScalarFormatter
plt.rcParams.update({'font.size': 14})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# we pick polatis, h100, 128 servers, 400G, 1port per NIC, plot the energy consumption of fat-tree, rail-optimized topology, and Opus

transceiver_power = 16 # https://www.fs.com/products/229253.html?now_cid=4089

# Using switch: NVIDIA Quantum-X800 XDR InfiniBand Switch, Q3400-RA
# https://www.naddod.com/products/nvidia-networking/102612?srsltid=AfmBOorimAH8n3NPPVieOwXdJaX0ZN2I9bUtPWQI-OUjHPMDnsrdjVk5
# https://docs.nvidia.com/networking/display/xdrswitcheshwum/specifications
# 7000 W with active cable
switch_port_power = 7000/144

ocs_port_power = 75/192 # polatis 6000N

transceiver_cost = 879 #https://www.fs.com/products/229253.html?now_cid=4089

# US$ 149939.00
switch_port_cost_fat_tree = 149939.00/144
switch_port_cost_rail_optimized = 149939.00/144

ocs_port_cost = 520

topologies = ['EPS Fat-tree', 'EPS Rail-Opt', 'Opus']

def get_cost_energy(total_server=128, num_gpu_per_server=8):

    total_gpu = total_server * num_gpu_per_server
    num_gpu_ports = total_server * num_gpu_per_server

    # Fat-tree
    k = int((total_gpu * 4) ** (1/3) + 0.5)
    radix = k

    num_switch_fat_tree = k ** 2 + k ** 2 // 4
    num_switch_ports_fat_tree = radix * num_switch_fat_tree
    print(f"Fat-tree: k={k}, num_switch_fat_tree={num_switch_fat_tree}, num_switch_ports_fat_tree={num_switch_ports_fat_tree}")

    # Rail-optimized
    # k = int((total_server * 4) ** (1/3) + 0.5)
    # radix = k
    # num_switch_fat_tree_per_rail = k ** 2
    # num_switch_spine = total_server # non-blocking

    # num_switch_rail_optimized = num_switch_fat_tree_per_rail * num_gpu_per_server + num_switch_spine
    # num_switch_ports_rail_optimized = radix * num_switch_fat_tree_per_rail * num_gpu_per_server + num_switch_spine * num_gpu_per_server
    k = int((total_server * 4) ** 0.5 + 0.5)
    num_switch_fat_tree_per_rail = total_server // (k // 2) * 2
    radix = k
    num_switch_spine = num_gpu_per_server * total_server // k
    num_switch_rail_optimized = num_switch_fat_tree_per_rail * num_gpu_per_server + num_switch_spine
    num_switch_ports_rail_optimized = radix * (num_switch_fat_tree_per_rail * num_gpu_per_server + num_switch_spine)
    print(f"Rail-optimized: k={k}, num_switch_rail_optimized={num_switch_rail_optimized}, num_switch_ports_rail_optimized={num_switch_ports_rail_optimized}")

    # opus
    num_ocs_opus = num_gpu_per_server
    radix = total_server * 2
    num_ocs_ports_opus = radix * num_ocs_opus


    # Cost
    total_cost_fat_tree = switch_port_cost_fat_tree * num_switch_ports_fat_tree + transceiver_cost * (num_gpu_ports)
    total_cost_rail_optimized = switch_port_cost_rail_optimized * num_switch_ports_rail_optimized + transceiver_cost * (num_gpu_ports)
    total_cost_opus = ocs_port_cost * num_ocs_ports_opus + transceiver_cost * num_gpu_ports

    # Power
    total_power_fat_tree = switch_port_power * num_switch_ports_fat_tree + transceiver_power * (num_gpu_ports )
    total_power_rail_optimized = switch_port_power * num_switch_ports_rail_optimized + transceiver_power * (num_gpu_ports)
    total_power_opus = ocs_port_power * num_ocs_ports_opus + transceiver_power * num_gpu_ports



    return [(total_cost_fat_tree, total_cost_rail_optimized, total_cost_opus),
            (total_power_fat_tree, total_power_rail_optimized, total_power_opus)]

num_servers = [8, 16, 32, 64]
num_gpu_per_server = 32

# Collect cost and power data for each topology across different server counts
costs_data = []
powers_data = []

# colors = ["orange","#d62728","#1f77b4"]

for total_server in num_servers:
    costs, powers = get_cost_energy(total_server=total_server, num_gpu_per_server=num_gpu_per_server)
    costs_data.append(costs)
    powers_data.append(powers)
    print(f"Cost saving: {costs[1] / costs[2]:.2f}x for Opus over Rail-optimized")
    print(f"Power saving: {powers[1] / powers[2]:.2f}x for Opus over Rail-optimized")

    print(f"\n !!Total servers: {total_server}")
    for i, topology in enumerate(topologies):
        print(f"{topology}: Cost = ${costs[i]:.2f}, Power = {powers[i]:.2f} W")
    print("-----------------------------------")

# Convert data to arrays for easier plotting
costs_data = np.array(costs_data)
powers_data = np.array(powers_data)

num_gpus = np.array(num_servers) * num_gpu_per_server

matplotlib.rcParams.update({"font.size": 14})

# Combine cost and power comparison plots vertically (stacked top/bottom)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

# Cost comparison bar plot
bar_width = 0.25
x_positions = np.arange(len(num_gpus))

for i, topology in enumerate(topologies):
    ax1.bar(x_positions + i * bar_width, costs_data[:, i]/1e6, bar_width, label=topology, color=colors[i])

# Add ratio labels on Opus bars (comparing to Rail-Opt)
# for j, x_pos in enumerate(x_positions):
#     ratio = costs_data[j, 2] / costs_data[j, 1]  # Opus / Rail-Opt
#     ax1.text(x_pos + 2 * bar_width, costs_data[j, 2]/1e6, f'{ratio:.1f}X',
#              ha='center', va='bottom', fontsize=10)

ax1.set_ylabel('Cost (Million $)')
ax1.set_xticks(x_positions + bar_width)

# Power comparison bar plot
for i, topology in enumerate(topologies):
    ax2.bar(x_positions + i * bar_width, powers_data[:, i]/1000, bar_width, label=topology, color=colors[i])

# # Add ratio labels on Opus bars (comparing to Rail-Opt)
# for j, x_pos in enumerate(x_positions):
#     ratio = powers_data[j, 2] / powers_data[j, 1]  # Opus / Rail-Opt
#     ax2.text(x_pos + 2 * bar_width, powers_data[j, 2]/1000, f'{ratio:.1f}X',
#              ha='center', va='bottom', fontsize=10)

ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Power (kW)')
ax2.set_xticklabels(num_gpus)

# Single shared legend inside the top subplot
ax1.legend(loc='upper left', ncol=1)

plt.tight_layout(h_pad=0.)
plt.show()
# Save the figure to a PDF file
fig.savefig("cost_energy_plots_GB200.pdf", format="pdf", bbox_inches='tight')