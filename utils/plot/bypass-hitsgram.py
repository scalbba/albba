import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter  # Make sure this import is included
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde

# File paths
data_file_path = '../../bfs/data/slice-bypass.csv'  # Path to the CSV file with the data
width_file_path = '../../bfs/data/column-bypass.csv'
smv_slice_path = '../../bfs/data/spmv-slice-bypass.csv'
x_axis_column_index = 0  # Column index for x-axis values, starting from 0

# # Read x-axis values from the specified column in the other CSV file
# x_axis_values = []
# with open(x_axis_file_path, newline='') as xfile:
#     reader = csv.reader(xfile)
#     for row in reader:
#         if len(row) > x_axis_column_index:  # Check if the column exists
#             x_axis_values.append(row[x_axis_column_index])
#         else:
#             x_axis_values.append("Missing Value")  # Placeholder if the column value is missing

slice_ratios = []  # To store the ratio for each row

with open(data_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Convert the 7th column and onwards to floats, ignoring non-numeric values
        numeric_data = [float(value) for value in row[6:] if value.replace('.', '', 1).isdigit()]
        
        # Ensure there's a value in the 6th column and it's numeric
        if row[5].replace('.', '', 1).isdigit():
            sixth_column_value = float(row[5])
            
            if numeric_data and sixth_column_value != 0:  # Avoid division by zero
                average = sum(numeric_data) / len(numeric_data)
                ratio = average / sixth_column_value
                if (ratio > 1):
                    print(ratio)
                slice_ratios.append(ratio)
            else:
                print(f"Row {reader.line_num}: No numeric data found from 7th column onwards or 6th column value is zero")
        else:
            print(f"Row {reader.line_num}: 6th column value is not numeric")


# print(slice_ratios)

column_ratios = []

with open(width_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Convert the 7th column and onwards to floats, ignoring non-numeric values
        numeric_data = [float(value) for value in row[6:] if value.replace('.', '', 1).isdigit()]
        
        # Ensure there's a value in the 6th column and it's numeric
        if row[5].replace('.', '', 1).isdigit():
            sixth_column_value = float(row[5])
            
            if numeric_data and sixth_column_value != 0:  # Avoid division by zero
                average = sum(numeric_data) / len(numeric_data)
                ratio = average / sixth_column_value
                column_ratios.append(ratio)
            else:
                print(f"Row {reader.line_num}: No numeric data found from 7th column onwards or 6th column value is zero")
        else:
            print(f"Row {reader.line_num}: 6th column value is not numeric")

spmv_slice_ratios=[]
with open(smv_slice_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Convert the 7th column and onwards to floats, ignoring non-numeric values
        numeric_data = [float(value) for value in row[6:] if value.replace('.', '', 1).isdigit()]
        
        # Ensure there's a value in the 6th column and it's numeric
        if row[5].replace('.', '', 1).isdigit():
            sixth_column_value = float(row[5])
            
            if numeric_data and sixth_column_value != 0:  # Avoid division by zero
                average = sum(numeric_data) / len(numeric_data)
                ratio = average / sixth_column_value
                spmv_slice_ratios.append(ratio)
            else:
                print(f"Row {reader.line_num}: No numeric data found from 7th column onwards or 6th column value is zero")
        else:
            print(f"Row {reader.line_num}: 6th column value is not numeric")

fig = plt.figure(figsize=(4.5, 7))
gs = gridspec.GridSpec(3, 1)

# Define a custom formatter for the y-axis labels
def custom_formatter(x, pos):
    """Formats the tick label to show percentage values."""
    return f'{x * 100:.0f}%'

n_ticks = 5
# Create histograms for each set of ratios
# Slice Ratios
ax1 = fig.add_subplot(gs[0, 0])
# For slice ratios
counts_slice, bin_edges_slice, _ = ax1.hist(slice_ratios, bins=10, color='#998ec3',rwidth=0.6,  edgecolor='#998ec3')
print("Slice Ratios:")
print("Counts:", counts_slice)
print("Bin edges:", bin_edges_slice)# ax1.set_title('Slice-Wise Ratios')
ax1.set_xlabel('(a) Intersection Bypass Ratio (slice-wise)', fontsize=12)
ax1.set_ylabel('#Matrices', fontsize=14)
ax1.yaxis.set_major_locator(MaxNLocator(n_ticks))
ax1.xaxis.set_major_locator(MaxNLocator(n_ticks))
ax1.grid(c='grey', alpha=0.8, linestyle='--')
ax1.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
mean_value = np.mean(slice_ratios) 
# Adding a vertical line for the mean
ax1.axvline(mean_value, color='r', linestyle='dashed', linewidth=2)

# Adding label for the mean line
ax1.text(mean_value, plt.ylim()[1] , f'Avg: {mean_value * 100:.2f}%', color='r', ha='center')
# print(counts)
# Column Ratios
# Fit KDE to the data
# Calculate the KDE
# data = slice_ratios
# kde = gaussian_kde(data)
# kde.set_bandwidth(bw_method='scott')  # 'scott' is a rule-of-thumb for bandwidth selection. You can also use a scalar or 'silverman'.

# # Generate values over the range of your data for plotting the KDE curve
# kde_x = np.linspace(min(data), max(data), 1000)
# kde_y = kde.evaluate(kde_x)  # Evaluate the density model at each point

# # Plot the KDE curve
# ax1.plot(kde_x, kde_y, color='red', linestyle='-', linewidth=2, label='KDE')


ax2 = fig.add_subplot(gs[1, 0])
densities_column, bin_edges_column, _ = ax2.hist(column_ratios, bins=10, rwidth=0.6,color='#67a9cf', edgecolor='#67a9cf')
print("\nColumn Ratios (Density):")
print("Densities:", densities_column)
print("Bin edges:", bin_edges_column)# ax2.set_title('Column-Wise Ratios')
ax2.set_xlabel('(b) Intersection Bypass Ratio (column-wise)', fontsize=12)
ax2.set_ylabel('#Matrices', fontsize=14)
ax2.yaxis.set_major_locator(MaxNLocator(n_ticks))
ax2.xaxis.set_major_locator(MaxNLocator(n_ticks))
ax2.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax2.grid(c='grey', alpha=0.8, linestyle='--')
mean_value = np.mean(column_ratios) 
# Adding a vertical line for the mean
ax2.axvline(mean_value, color='r', linestyle='dashed', linewidth=2)

# Adding label for the mean line
ax2.text(mean_value, plt.ylim()[1] , f'Avg: {mean_value * 100:.2f}%', color='r', ha='center')

# SPMV Slice Ratios
ax3 = fig.add_subplot(gs[2, 0])
densities_spmv, bin_edges_spmv, _ = ax3.hist(spmv_slice_ratios, bins=10,rwidth=0.6, color='#7fbf7b', edgecolor='#7fbf7b')
print("\nSPMV Slice Ratios (Density):")
print("Densities:", densities_spmv)
print("Bin edges:", bin_edges_spmv)# ax3.set_title('SPMV Slice Ratios')
ax3.set_xlabel('(c) Visited Bypass Ratio', fontsize=12)
ax3.set_ylabel('#Matrices', fontsize=14)
ax3.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax3.grid(c='grey', alpha=0.8, linestyle='--')
mean_value = np.mean(spmv_slice_ratios) 
# Adding a vertical line for the mean
# ax3.set_xlim(0, 1) 
ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax3.axvline(mean_value, color='r', linestyle='dashed', linewidth=2)
ax3.yaxis.set_major_locator(MaxNLocator(n_ticks))
ax3.xaxis.set_major_locator(MaxNLocator(n_ticks))
# Adding label for the mean line
ax3.text(mean_value, plt.ylim()[1] , f'Avg: {mean_value * 100:.2f}%', color='r', ha='center')

# n_ticks = 10
# plt.gca().xaxis.set_major_locator(MaxNLocator(n_ticks))

plt.tight_layout()
plt.savefig("../../bfs/figures/bypass_histogram.eps", dpi=300)
plt.show()
