import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Read the data
df = pd.read_csv('jira_dataset.csv')

# 1. Data Analysis and Visualization
print("=== Initial Data Analysis ===")
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# Create visualizations
plt.figure(figsize=(15, 10))

# Priority Distribution
plt.subplot(2, 2, 1)
df['priority'].value_counts().plot(kind='bar')
plt.title('Task Priority Distribution')
plt.xticks(rotation=45)

# Deadline Distribution
plt.subplot(2, 2, 2)
df['task_deadline'].value_counts().plot(kind='bar')
plt.title('Task Deadline Distribution')
plt.xticks(rotation=45)

# Top 10 Assignees
plt.subplot(2, 2, 3)
df['task_assignee'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Task Assignees')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('task_distributions.png')
plt.close()

# 2. Data Cleaning
print("\n=== Data Cleaning ===")

# Remove duplicates
df_cleaned = df.drop_duplicates()
print(f"Removed {len(df) - len(df_cleaned)} duplicate records")

# Fill missing values
df_cleaned['status'] = df_cleaned['status'].fillna('unknown')
df_cleaned['clean_summary'] = df_cleaned['clean_summary'].fillna('')

# Standardize column names
df_cleaned.columns = [col.lower().replace(' ', '_') for col in df_cleaned.columns]

# 3. Text Preprocessing
print("\n=== Text Preprocessing ===")

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error processing text: {e}")
        return ''

# Apply preprocessing to description
df_cleaned['processed_description'] = df_cleaned['project_description'].apply(preprocess_text)

# Save cleaned dataset
df_cleaned.to_csv('cleaned_tasks.csv', index=False)

print("\nCleaned dataset saved to 'cleaned_tasks.csv'")
print("\nSample of processed descriptions:")
print(df_cleaned[['project_description', 'processed_description']].head()) 