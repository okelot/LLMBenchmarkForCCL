import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Load CSV file
file_path = 'results/evaluated_case_model_results_with_section_similarity.csv'
df = pd.read_csv(file_path)

# Define similarity columns
similarity_columns = [
    'facts_similarity',
    'issue_similarity',
    'decision_similarity',
    'reasons_similarity',
    'ratio_similarity'
]

# Clean data: drop rows with any missing similarity scores
df_clean = df.dropna(subset=similarity_columns)

# Compute average similarity scores per model
average_scores = df_clean.groupby('Model_ID')[similarity_columns].mean().reset_index()

# Melt DataFrame for plotting
melted_df = average_scores.melt(id_vars='Model_ID', var_name='Section', value_name='Average Similarity')

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(data=melted_df, x='Model_ID', y='Average Similarity', hue='Section')
plt.xticks(rotation=45, ha='right')
plt.title('Average Section Similarity Scores by Model')
plt.tight_layout()
plt.legend(title='Section')
plt.grid(True)

# Create output directory if not exists
output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

# Generate timestamped filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'{output_dir}/model_similarity_chart_{timestamp}.png'

# Save the plot
plt.savefig(filename)
plt.close()

print(f'Chart saved to: {filename}')
