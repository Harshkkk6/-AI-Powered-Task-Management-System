import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create the cells
cells = [
    nbf.v4.new_markdown_cell("# AI-Powered Task Management System\n\nThis notebook implements an AI-powered task management system using NLP and ML techniques to analyze and process task data from Jira."),
    
    nbf.v4.new_markdown_cell("## 1. Data Loading and Initial Setup"),
    
    nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from IPython.display import display

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette('husl')"""),
    
    nbf.v4.new_markdown_cell("## 2. Load and Clean Dataset"),
    
    nbf.v4.new_code_cell("""# Load the dataset
try:
    df = pd.read_csv('jira_dataset.csv')
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: File 'jira_dataset.csv' not found")
except Exception as e:
    print(f"Error loading data: {str(e)}")

# Display first few rows
print("\\nFirst few rows of the dataset:")
display(df.head())"""),
    
    nbf.v4.new_markdown_cell("## 3. Data Preprocessing"),
    
    nbf.v4.new_code_cell("""# Drop rows where clean_summary is null
print(f"Number of rows before dropping nulls: {df.shape[0]}")
df = df.dropna(subset=['clean_summary'])
print(f"Number of rows after dropping nulls: {df.shape[0]}")

# Standardize deadline entries
def standardize_deadline(text):
    if pd.isna(text):
        return text
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Common replacements
    replacements = {
        'tow': 'two',
        'for days': 'four days',
        'tree': 'three',
        'won': 'one',
        'to': 'two',
        'free': 'three'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

# Apply standardization to deadline column
if 'deadline' in df.columns:
    df['deadline'] = df['deadline'].apply(standardize_deadline)
    print("\\nSample of standardized deadlines:")
    display(df['deadline'].head())"""),
    
    nbf.v4.new_markdown_cell("## 4. Text Preprocessing"),
    
    nbf.v4.new_code_cell("""def preprocess_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into string
    return ' '.join(tokens)

# Apply preprocessing to clean_summary column
if 'clean_summary' in df.columns:
    df['processed_summary'] = df['clean_summary'].apply(preprocess_text)
    
    # Display sample of original and processed summaries
    print("Sample of original and processed summaries:")
    display(pd.DataFrame({
        'Original': df['clean_summary'].head(),
        'Processed': df['processed_summary'].head()
    }))"""),
    
    nbf.v4.new_markdown_cell("## 5. Exploratory Data Analysis (EDA)"),
    
    nbf.v4.new_code_cell("""# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot issue type distribution
if 'issue_type' in df.columns:
    sns.countplot(data=df, y='issue_type', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Issue Types')

# Plot priority distribution
if 'priority' in df.columns:
    sns.countplot(data=df, x='priority', ax=axes[0,1])
    axes[0,1].set_title('Distribution of Priorities')
    axes[0,1].tick_params(axis='x', rotation=45)

# Plot task assignee distribution (top 10)
if 'task_assignee' in df.columns:
    top_assignees = df['task_assignee'].value_counts().head(10)
    sns.barplot(x=top_assignees.values, y=top_assignees.index, ax=axes[1,0])
    axes[1,0].set_title('Top 10 Task Assignees')

# Plot summary statistics
if 'processed_summary' in df.columns:
    word_counts = df['processed_summary'].str.split().str.len()
    sns.histplot(data=word_counts, bins=30, ax=axes[1,1])
    axes[1,1].set_title('Distribution of Word Counts in Processed Summaries')
    axes[1,1].set_xlabel('Number of Words')

plt.tight_layout()
plt.show()

# Display summary statistics
print("\\nSummary Statistics:")
display(df.describe())

# Display value counts for categorical columns
print("\\nValue counts for categorical columns:")
for col in ['issue_type', 'priority', 'task_assignee']:
    if col in df.columns:
        print(f"\\n{col}:")
        display(df[col].value_counts())"""),
    
    nbf.v4.new_markdown_cell("## 6. Model Training and Evaluation"),
    
    nbf.v4.new_code_cell("""# Import required libraries for model training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import seaborn as sns

# Prepare the data
X = df['processed_summary']
y = df['issue_type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Data split and vectorization completed:")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")"""),
    
    nbf.v4.new_markdown_cell("### Train and Evaluate Models"),
    
    nbf.v4.new_code_cell("""# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print(f"\\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return accuracy, precision, recall, f1

# Train and evaluate Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_metrics = evaluate_model(nb_model, X_test_tfidf, y_test, 'Naive Bayes')

# Train and evaluate LinearSVC
svc_model = LinearSVC(random_state=42)
svc_model.fit(X_train_tfidf, y_train)
svc_metrics = evaluate_model(svc_model, X_test_tfidf, y_test, 'LinearSVC')"""),
    
    nbf.v4.new_markdown_cell("### Save Best Model"),
    
    nbf.v4.new_code_cell("""# Compare models and save the best one
nb_f1 = nb_metrics[3]
svc_f1 = svc_metrics[3]

if nb_f1 > svc_f1:
    best_model = nb_model
    best_model_name = 'Naive Bayes'
else:
    best_model = svc_model
    best_model_name = 'LinearSVC'

# Save the best model and vectorizer
joblib.dump(best_model, 'task_classifier.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print(f"\\nBest model ({best_model_name}) saved as 'task_classifier.pkl'")
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")"""),
    
    nbf.v4.new_markdown_cell("## 7. Priority Prediction and Workload Analysis"),
    
    nbf.v4.new_markdown_cell("### Priority Prediction Model"),
    
    nbf.v4.new_code_cell("""# Import required libraries for priority prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Prepare the data for priority prediction
X_priority = df['processed_summary']
y_priority = df['priority']

# Split the data
X_train_priority, X_test_priority, y_train_priority, y_test_priority = train_test_split(
    X_priority, y_priority, test_size=0.2, random_state=42
)

# Create and fit TF-IDF vectorizer for priority prediction
tfidf_priority = TfidfVectorizer(max_features=5000)
X_train_priority_tfidf = tfidf_priority.fit_transform(X_train_priority)
X_test_priority_tfidf = tfidf_priority.transform(X_test_priority)

print("Data prepared for priority prediction:")
print(f"Training set size: {X_train_priority.shape[0]}")
print(f"Test set size: {X_test_priority.shape[0]}")"""),
    
    nbf.v4.new_markdown_cell("### Model Training with GridSearchCV"),
    
    nbf.v4.new_code_cell("""# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create and train Random Forest with GridSearchCV
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(X_train_priority_tfidf, y_train_priority)

# Print best parameters
print("Best Random Forest parameters:")
print(rf_grid.best_params_)

# Get best model
best_rf_model = rf_grid.best_estimator_

# Make predictions
y_pred_priority = best_rf_model.predict(X_test_priority_tfidf)

# Print classification report
print("\\nClassification Report:")
print(classification_report(y_test_priority, y_pred_priority))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_priority, y_pred_priority)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Priority Prediction')
plt.xlabel('Predicted Priority')
plt.ylabel('Actual Priority')
plt.show()

# Save the model and vectorizer
joblib.dump(best_rf_model, 'priority_predictor.pkl')
joblib.dump(tfidf_priority, 'priority_tfidf_vectorizer.pkl')
print("\\nModel and vectorizer saved as 'priority_predictor.pkl' and 'priority_tfidf_vectorizer.pkl'")"""),
    
    nbf.v4.new_markdown_cell("### Workload Analysis"),
    
    nbf.v4.new_code_cell("""# Calculate tasks per assignee
tasks_per_assignee = df['task_assignee'].value_counts()

# Create figure for workload visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=tasks_per_assignee.index, y=tasks_per_assignee.values)
plt.title('Number of Tasks per Assignee')
plt.xlabel('Assignee')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Identify assignee with least workload
min_workload_assignee = tasks_per_assignee.idxmin()
min_workload = tasks_per_assignee.min()

print(f"\\nAssignee with least workload: {min_workload_assignee}")
print(f"Number of tasks: {min_workload}")

# Calculate workload statistics
print("\\nWorkload Statistics:")
print(f"Average tasks per assignee: {tasks_per_assignee.mean():.2f}")
print(f"Median tasks per assignee: {tasks_per_assignee.median():.2f}")
print(f"Maximum tasks per assignee: {tasks_per_assignee.max()}")

# Create workload distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=tasks_per_assignee.values, bins=20)
plt.title('Distribution of Workload per Assignee')
plt.xlabel('Number of Tasks')
plt.ylabel('Number of Assignees')
plt.show()"""),
    
    nbf.v4.new_markdown_cell("### Workload Balance Recommendations"),
    
    nbf.v4.new_code_cell("""# Calculate workload imbalance
workload_imbalance = tasks_per_assignee.max() - tasks_per_assignee.min()
print(f"Workload imbalance (max - min tasks): {workload_imbalance}")

# Identify overloaded assignees (more than 1.5 times the median)
median_tasks = tasks_per_assignee.median()
overloaded_assignees = tasks_per_assignee[tasks_per_assignee > (1.5 * median_tasks)]
print("\\nOverloaded assignees (more than 1.5 times median workload):")
for assignee, tasks in overloaded_assignees.items():
    print(f"{assignee}: {tasks} tasks")

# Recommend task redistribution
print("\\nTask Redistribution Recommendations:")
print(f"1. Consider reassigning tasks from overloaded assignees to {min_workload_assignee}")
print("2. Target workload per assignee should be around the median:", f"{median_tasks:.1f} tasks")
print("3. Current workload imbalance:", f"{workload_imbalance} tasks")"""),
    
    nbf.v4.new_markdown_cell("## 8. Deploying the Model with Streamlit"),
    
    nbf.v4.new_markdown_cell("### Running the Streamlit App"),
    
    nbf.v4.new_code_cell("""# Install required packages
!pip install streamlit pandas matplotlib seaborn scikit-learn joblib"""),
    
    nbf.v4.new_markdown_cell("### Streamlit App Instructions"),
    
    nbf.v4.new_markdown_cell("""To run the Streamlit app:

1. Ensure all required files are present:
   - streamlit_app.py
   - task_classifier.pkl
   - priority_predictor.pkl
   - tfidf_vectorizer.pkl
   - priority_tfidf_vectorizer.pkl
   - jira_dataset.csv

2. Open a terminal and navigate to the project directory

3. Run the app with:
   ```bash
   streamlit run streamlit_app.py
   ```

4. The app will open in your default web browser

The Streamlit app provides:
- A text input area for task descriptions
- Automatic prediction of issue type and priority
- Real-time workload visualization
- Workload statistics and recommendations"""),
    
    nbf.v4.new_markdown_cell("### App Features"),
    
    nbf.v4.new_markdown_cell("""The Streamlit app includes:

1. **Task Prediction**
   - Enter task description
   - Get predicted issue type and priority
   - Real-time predictions

2. **Workload Visualization**
   - Bar chart of tasks per assignee
   - Workload statistics
   - Team workload distribution

3. **User Interface**
   - Clean, modern design
   - Responsive layout
   - Clear instructions
   - Error handling""")
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('task_management_project.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 