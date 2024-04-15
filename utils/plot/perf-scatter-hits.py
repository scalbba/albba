import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gmean

columns = ['name', 'rows', 'columns', 'entries', 'iter', 'albba']

file_path2 = '../../bfs/data/BFS_DFC_ompruntime-results.csv'
df_h100 = pd.read_csv(file_path2, header=None, names=columns)

# Ensuring all necessary columns are present and not empty
df_h100_filtered = df_h100[['rows','entries', 'iter', 'albba']].dropna()

# Calculating speedups
# df_h100_filtered['TEPS_graphblas'] = df_h100_filtered['entries']/df_h100_filtered['graphblast'] /1000000
df_h100_filtered['TEPS_albba'] = df_h100_filtered['entries']/df_h100_filtered['albba']/1000000


# df_h100_filtered['TEPS_graphblas'] = np.log10(df_h100_filtered['TEPS_graphblas'])
df_h100_filtered['TEPS_albba'] = np.log10(df_h100_filtered['TEPS_albba'])


# df_h100_filtered['speedup_graphblast'] =  df_h100_filtered['graphblast']/df_h100_filtered['albba']
# df_h100_filtered['speedup_enterprise'] = df_h100_filtered['enterprise']/df_h100_filtered['albba']

# Calculating log10 of entries
df_h100_filtered['log_entries'] = np.log10(df_h100_filtered['entries'])
df_h100_filtered['log_iters'] = np.log10(df_h100_filtered['iter'])


# Setting up the figure and GridSpec layout
fig = plt.figure(figsize=(10,3.2))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # Adjust width_ratios and height_ratios for desired subplot sizes

# Creating the subplots
ax2 = fig.add_subplot(gs[0, 0])
ax4 = fig.add_subplot(gs[0, 1])

# Setting up the plot
labels = ['ALBBA (this work)']
markers = [ 'x']
scatter_size  = [10, 20]


colors = ['#E53935']

for i, column in enumerate(['TEPS_albba']):
    ax2.scatter(df_h100_filtered['log_iters'], df_h100_filtered[column], s=scatter_size[i], marker=markers[i], color=colors[i], label=labels[i])

# Improving plot aesthetics
# ax2.set_yscale('log')
ax2.set_ylabel('Performance (GTEPS) \n ($\log_{10}$ scale)', fontsize=13)
# ax2.text(-0.3, -0.6, 'Performance (GTEPS)', fontsize=13, va='center', ha='center', rotation='vertical')
# ax2.text(-0.15,-0.6, '($\log_{10}$ scale)', fontsize=11, va='center', ha='center', rotation='vertical')
ax2.set_xlabel('#Iterations ($\log_{10}$ scale)', fontsize=13)
ax2.grid(c='grey', alpha=0.8, linestyle='--')
ax2.legend()
n_ticks = 5


# ax4.hist(np.log10(df_h100_filtered['speedup_graphblast']), bins=30,color='#1b9e77')

# Improving plot aesthetics
# ax4.set_xlabel('Speedup over BraphBLAST \n ($\log_{10}$ scale)', fontsize=11)
# ax4.set_ylabel('#Matrices', fontsize=14)
# plt.gca().yaxis.set_major_locator(MaxNLocator(n_ticks))
# plt.gca().xaxis.set_major_locator(MaxNLocator(n_ticks))

# ax4.grid(c='grey', alpha=0.8, linestyle='--')


# log_mean = np.log(df_h100_filtered['speedup_graphblast']).mean()
# geometric_mean = np.exp(log_mean)
# print(np.log10(geometric_mean))
# ax4.axvline(np.log10(geometric_mean), color='r', linestyle='dashed', linewidth=2)
# ax4.text(np.log10(geometric_mean), plt.ylim()[1] , f'Geo mean: {geometric_mean:.2f}x',fontsize=12, color='r', ha='center')


plt.tight_layout()

plt.savefig("../../bfs/figures/perf-hist.eps", dpi = 300)

plt.show()
