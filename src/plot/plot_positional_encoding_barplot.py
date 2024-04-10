import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data in Pandas DataFrame format
data = {
    'Framework': ['With', 'Without'],
    'ACIDS Acc': [0.9265, 0.9365],
    'ACIDS F1': [0.7596, 0.7919],
    'PAMAP2 Acc': [0.8312, 0.8420],
    'PAMAP2 F1': [0.8120, 0.8205],
    'HAR Acc': [0.8407, 0.9250],
    'HAR F1': [0.8567, 0.9327],
    'Parkland Acc': [None, 0.9524],
    'Parkland F1': [None, 0.9514]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# Use the academic-style font
plt.rcParams['font.family'] = 'serif'

# Set the color palette
colors = sns.color_palette("pastel", n_colors=len(df.columns) - 1)

# Plot each metric for frameworks
for idx, column in enumerate(df.columns[1:], 1):
    sns.barplot(x='Framework', y=column, data=df, color=colors[idx-1], label=column)

plt.legend(loc='upper left', bbox_to_anchor=(1,1), title="Metrics")
plt.title("Effect of positional encoding on frameworks")
plt.ylabel("Score")
plt.tight_layout()

# Save the plot
plt.savefig('result/figures/positional_encoding_effect.pdf', bbox_inches='tight')
