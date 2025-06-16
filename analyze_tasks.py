import pandas as pd

# Load the CSV file
df = pd.read_csv('jira_dataset.csv')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Display basic information about the dataset
print("\nDataset information:")
print(df.info()) 