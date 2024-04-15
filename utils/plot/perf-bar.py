import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
csv_file_path = 'NEC20/12_matrices.csv'
data = pd.read_csv(csv_file_path, header=None, nrows=12)

def safe_divide(numerator, denominator):
    return np.where(denominator != 0, numerator / denominator, 0)

# Calculate the y-values for the bars, adjusting for division by zero
y_values_1 = safe_divide(data.iloc[:, 5], data.iloc[:, 7]) / 1000000  # 4th divided by 8th column, GraphBLAST
y_values_2 = safe_divide(data.iloc[:, 5], data.iloc[:, 8]) / 1000000  # 4th divided by 9th column, ALBBA
y_values_3 = safe_divide(data.iloc[:, 5], data.iloc[:, 9]) / 1000000  # 4th divided by 10th column, Enterprise


labels = ['Hamrle','amazon','cant','net150',
          'dblp','mc2depi','roadNet','citation',
          'ML_Geer' ,'twitter','vsp','wiki-2007'
          ]
        #   'net', 'benelechi','ex35', 'twitter',
        #   'Citeseer','exdata','bundle'

# Number of bars/groups
N = len(labels)
# len(data)

# Setup the figure and axes
fig, ax = plt.subplots(figsize=(8, 4))

# Create an index for each tick position
ind = np.arange(N)  
width = 0.25  # Width of the bars

# Plot the bars
bars1 = ax.bar(ind - width/2, y_values_1, width, color= '#9ecae1' ,label='GraphBLAST')
bars2 = ax.bar(ind + width/2 , y_values_3, width, color = '#4292c6', label='Enterprise')
bars3 = ax.bar(ind + width + width/2 , y_values_2, width, color = '#084594', label='ALBBA (this work)')

# Function to add labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=10)

# Add some text for labels, title, and axes ticks
ax.set_xlabel('Matrices', fontsize=18)
ax.set_ylabel('Performance (GTEPS)', fontsize=18)
ax.grid(c='grey', alpha=0.8, linestyle='--')

# ax.set_ylim(0, 10)
# ax.set_title('Values of 4th column divided by 8th and 9th columns')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(labels, rotation = 25, fontsize=11)  # Assuming you want row indices as labels

# Call the function to add labels on top of each bar
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

fig.subplots_adjust(bottom=0.14)  # Make room for the legend
# Add a legend
ax.legend( loc='upper left',fontsize=10)

plt.savefig("perf-bar2-.eps", dpi = 300)

# Display the plot
plt.show()
