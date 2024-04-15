import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
def read_specific_line(file_path, line_number):
    with open(file_path, 'r') as file:
        for current_line_number, line in enumerate(file, start=1):
            if current_line_number == line_number:
                return line.strip().split(',')[6:]  # Adjust if different columns are needed
    return None

def find_row_max_values_across_files(files, num_rows):
    max_values = [0] * num_rows
    for file_path in files:
        with open(file_path, 'r') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number <= num_rows:
                    values = line.strip().split(',')[6:]  # Assuming data starts from the 7th column
                    if values:
                        row_max = max([float(value) for value in values])
                        max_values[current_line_number-1] = max(max_values[current_line_number-1], row_max)
                else:
                    break
    return max_values

# Paths to your CSV files
file_paths = [
    'NEC20/kernel-selection-spmv-nobypass.csv',
    'NEC20/ker-selection-merge.csv',
    'NEC20/kernel-selection-spmv.csv',
    'NEC20/kernel-selection-spmspv.csv', 
]

labels = ['SpMV (classic row-major SELL-C-$\\sigma$)', 
          'SpMSpV (classic column-major SELL-C-$\\sigma$)', 
          'SpMV (SELL-C-$\\sigma$ with bypass, this work)', 
          'SpMSpV (SELL-C-$\\sigma$ with bypass, this work)']

colors = ['blue', 'green', 'red', 'purple']
line_styles = ['-', '--', '-', '--']
markers = ['o', 's', 'x', 'p']
marker_sizes = [4, 4, 0.3, 4, 4, 0.3, 0.3, 0.3]  # Adjust the marker size as needed
line = []
title = ['Hamrle','amazon','cant','dblp',
         'citation','mc2depi','roadNet','wiki-2007',
         'vsp','wiki']
n_ticks = 4

# Focus on the last two files for maximum y-values
last_two_files = file_paths[-2:]  # Get the last two file paths
num_subplots = 8  # Assuming 8 subfigures corresponding to the first 8 rows
row_max_values = find_row_max_values_across_files(last_two_files, num_subplots)

fig, axs = plt.subplots(2, 4, figsize=(10, 4))  # Adjust figure size as needed
axs = axs.flatten()

for subplot_idx in range(num_subplots):
    for file_idx, file_path in enumerate(file_paths):
        line_data = read_specific_line(file_path, subplot_idx + 1)  # Line numbers start at 1
        if line_data:
            float_data = [float(value) for value in line_data]
            x_values = range(len(float_data))
            axs[subplot_idx].plot(x_values, np.log10(float_data), color=colors[file_idx], marker=markers[file_idx], linestyle=line_styles[file_idx], linewidth=0.5, markersize=marker_sizes[subplot_idx], label=labels[file_idx] if subplot_idx == 0 else "")
    
    # axs[subplot_idx].set_ylim(0, row_max_values[subplot_idx] * 1.1)  # Setting y-axis limit with a small margin
    axs[subplot_idx].yaxis.set_major_locator(MaxNLocator(n_ticks))
    # axs[subplot_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[subplot_idx].set_xlabel('Iterations')
    axs[subplot_idx].set_ylabel('$\log_{10}$ runtime')
    axs[subplot_idx].set_title(title[subplot_idx ])


for ax in axs:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.subplots_adjust(top=0.85)  # Make room for the legend

# Create a single legend for all subplots at the upper center
# Adjust bbox_to_anchor as needed if the legend is not correctly positioned
fig.legend(labels, loc='upper center', bbox_to_anchor=(0, 0.95, 1, 0.04), ncol=2, fontsize=10, borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the bottom parameter if necessary

plt.savefig("kernels-steptime-2.eps", dpi = 300)


plt.show()